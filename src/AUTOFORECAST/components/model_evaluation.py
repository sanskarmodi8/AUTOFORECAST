from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
    MeanSquaredError,
    MeanSquaredScaledError,
    MedianAbsoluteError,
    MedianSquaredError,
)
from sktime.split import ExpandingWindowSplitter

from AUTOFORECAST import logger
from AUTOFORECAST.constants import AVAIL_METRICS
from AUTOFORECAST.entity.config_entity import ModelEvaluationConfig
from AUTOFORECAST.utils.common import load_bin, save_json


class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, config: ModelEvaluationConfig):
        pass


class UnivariateEvaluationStrategy(ModelEvaluationStrategy):
    def _validate_and_prepare_data(self, y):
        """Validate and prepare the time series data."""
        if y is None or len(y) == 0:
            raise ValueError("Input time series is empty or None")

        # Convert to pandas Series if DataFrame
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Input must be univariate (single column)")
            y = y.iloc[:, 0]

        # Remove any infinite values
        y = y.replace([np.inf, -np.inf], np.nan)

        # Ensure index is datetime
        if not isinstance(y.index, pd.DatetimeIndex):
            y.index = pd.to_datetime(y.index)

        return y

    def _load_data(self, config):
        """Load test data and model"""
        try:
            y = pd.read_csv(
                Path(config.test_data_dir) / "y.csv", index_col=0, parse_dates=True
            )
            y = self._validate_and_prepare_data(y)
            model = load_bin(Path(config.model))
            return y, model
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _create_plots(self, results, config):
        """Create and save evaluation plots using plotly"""
        try:
            # Get actual and predicted values
            y_true = results.y_test.values.flatten()
            y_pred = results.y_pred.values.flatten()
            time_index = results.y_test.index

            # Forecast vs Actual Plot
            fig_forecast = go.Figure()

            fig_forecast.add_trace(
                go.Scatter(
                    x=time_index,
                    y=y_true,
                    name="Actual",
                    mode="lines+markers",
                    line=dict(color="blue"),
                )
            )

            fig_forecast.add_trace(
                go.Scatter(
                    x=time_index,
                    y=y_pred,
                    name="Forecast",
                    mode="lines+markers",
                    line=dict(color="red", dash="dash"),
                )
            )

            fig_forecast.update_layout(
                title="Forecast vs Actual Values",
                xaxis_title="Time",
                yaxis_title="Value",
                hovermode="x unified",
                showlegend=True,
            )

            fig_forecast.write_html(
                str(config.forecast_vs_actual_plot).replace(".png", ".html")
            )

            # Residuals Analysis
            residuals = y_true - y_pred

            fig_residuals = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Residuals Over Time", "Residuals Distribution"),
            )

            fig_residuals.add_trace(
                go.Scatter(
                    x=time_index,
                    y=residuals,
                    mode="lines+markers",
                    name="Residuals",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )

            fig_residuals.add_hline(
                y=0, line_dash="dash", line_color="red", row=1, col=1
            )

            fig_residuals.add_trace(
                go.Histogram(x=residuals, nbinsx=30, name="Distribution"), row=2, col=1
            )

            fig_residuals.update_layout(
                title="Residuals Analysis", showlegend=False, height=800
            )

            fig_residuals.write_html(
                str(config.forecast_vs_actual_plot).replace(".png", "_residuals.html")
            )

        except Exception as e:
            logger.error(f"Error creating plots: {str(e)}")
            raise

    def _validate_and_prepare_data(self, y):
        """Validate and prepare the time series data with additional checks."""
        if y is None or len(y) == 0:
            raise ValueError("Input time series is empty or None")

        # Convert to pandas Series if DataFrame
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Input must be univariate (single column)")
            y = y.iloc[:, 0]

        # Handle non-positive values (important for log transformation)
        min_val = y.min()
        if min_val <= 0:
            logger.warning(
                f"Data contains non-positive values. Minimum value: {min_val}"
            )
            # Add a constant to make all values positive
            offset = abs(min_val) + 1
            y = y + offset
            logger.info(f"Added offset of {offset} to make all values positive")

        # Remove any infinite values
        y = y.replace([np.inf, -np.inf], np.nan)

        # Handle missing values
        if y.isna().any():
            logger.warning(f"Found {y.isna().sum()} missing values")
            # Forward fill, then backward fill any remaining
            y = y.fillna(method="ffill").fillna(method="bfill")

        # Ensure index is datetime
        if not isinstance(y.index, pd.DatetimeIndex):
            y.index = pd.to_datetime(y.index)

        return y

    def evaluate(self, config):
        """Run evaluation for univariate forecasting with improved error handling"""
        logger.info("Running univariate evaluation")

        try:
            # Load and prepare data
            y, model = self._load_data(config)
            y = self._validate_and_prepare_data(y)

            # Calculate forecast horizon
            fh_length = max(int(len(y) * 0.2), 1)
            fh = ForecastingHorizon(np.arange(1, fh_length + 1), is_relative=True)

            # Setup CV splitter
            initial_window = max(int(len(y) * 0.6), 10)  # Ensure enough training data
            if initial_window >= len(y):
                raise ValueError(
                    f"Initial window size ({initial_window}) must be less than data length ({len(y)})"
                )

            cv = ExpandingWindowSplitter(
                initial_window=initial_window, step_length=1, fh=fh
            )

            # Create scoring metrics dictionary
            scoring = {
                "MeanAbsoluteError": MeanAbsoluteError(),
                "MeanSquaredError": MeanSquaredError(),
                "RootMeanSquaredError": MeanSquaredError(square_root=True),
                # Use symmetric MAPE which handles zeros better
                "MeanAbsolutePercentageError": MeanAbsolutePercentageError(
                    symmetric=True
                ),
                "MedianAbsoluteError": MedianAbsoluteError(),
            }

            scores = {}
            last_results = None

            for metric in config.chosen_metrics:
                logger.info(f"Evaluating using metric: {metric}")
                try:
                    # Set error_score='raise' to get detailed error messages during evaluation
                    results = evaluate(
                        forecaster=model,
                        y=y,
                        cv=cv,
                        scoring=scoring[metric],
                        error_score="raise",
                        return_data=True,
                    )

                    # Store results for plotting
                    last_results = results

                    # Calculate metric score
                    metric_col = (
                        f"test_{metric}"
                        if metric != "RootMeanSquaredError"
                        else "test_MeanSquaredError"
                    )
                    metric_scores = results.loc[:, metric_col]

                    # Handle case where all scores are NaN
                    if metric_scores.isna().all():
                        logger.warning(f"All scores for {metric} are NaN. Skipping...")
                        continue

                    # Calculate final score
                    mean_score = metric_scores.mean()
                    if metric == "RootMeanSquaredError":
                        mean_score = np.sqrt(mean_score)

                    scores[metric] = float(mean_score)
                    logger.info(f"{metric}: {mean_score}")

                except Exception as metric_error:
                    logger.error(f"Error calculating {metric}: {str(metric_error)}")
                    continue

            if not scores:
                raise ValueError(
                    "No valid metrics were calculated. Check if your model's transformers (especially LogTransformer) are compatible with your data."
                )

            # Save metrics
            save_json(Path(config.scores), scores)

            # Create plots only if we have valid results
            if last_results is not None:
                self._create_plots(last_results, config)

            return scores

        except Exception as e:
            logger.error(f"Error in univariate evaluation: {str(e)}")
            raise


class MultivariateEvaluationStrategy(ModelEvaluationStrategy):
    def _load_data(self, config):
        """Load test data and model"""
        try:
            y = pd.read_csv(
                Path(config.test_data_dir) / "y.csv", index_col=0, parse_dates=True
            )
            X = pd.read_csv(
                Path(config.test_data_dir) / "X.csv", index_col=0, parse_dates=True
            )
            model = load_bin(Path(config.model))
            return y, X, model
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _create_plots(self, results, X, config):
        """Create and save evaluation plots using plotly"""
        try:
            # Get actual and predicted values
            y_true = results.y_test.values.flatten()
            y_pred = results.y_pred.values.flatten()
            time_index = results.y_test.index

            # Forecast vs Actual Plot
            fig_forecast = go.Figure()

            fig_forecast.add_trace(
                go.Scatter(
                    x=time_index,
                    y=y_true,
                    name="Actual",
                    mode="lines+markers",
                    line=dict(color="blue"),
                )
            )

            fig_forecast.add_trace(
                go.Scatter(
                    x=time_index,
                    y=y_pred,
                    name="Forecast",
                    mode="lines+markers",
                    line=dict(color="red", dash="dash"),
                )
            )

            fig_forecast.update_layout(
                title="Forecast vs Actual Values (with Exogenous Variables)",
                xaxis_title="Time",
                yaxis_title="Value",
                hovermode="x unified",
                showlegend=True,
            )

            fig_forecast.write_html(
                str(config.forecast_vs_actual_plot).replace(".png", ".html")
            )

            # Enhanced residuals analysis for multivariate
            residuals = y_true - y_pred

            fig_residuals = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Residuals Over Time",
                    "Residuals Distribution",
                    "Residuals vs Predicted",
                    "Feature Importance",
                ),
            )

            fig_residuals.add_trace(
                go.Scatter(
                    x=time_index,
                    y=residuals,
                    mode="lines+markers",
                    name="Residuals",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )

            fig_residuals.add_hline(
                y=0, line_dash="dash", line_color="red", row=1, col=1
            )

            fig_residuals.add_trace(
                go.Histogram(x=residuals, nbinsx=30, name="Distribution"), row=1, col=2
            )

            fig_residuals.add_trace(
                go.Scatter(
                    x=y_pred, y=residuals, mode="markers", name="Residuals vs Predicted"
                ),
                row=2,
                col=1,
            )

            # Feature correlations with residuals
            X_subset = X.loc[time_index]  # Align X with the evaluation period
            correlations = X_subset.corrwith(pd.Series(residuals, index=time_index))

            fig_residuals.add_trace(
                go.Bar(
                    x=correlations.index,
                    y=correlations.values,
                    name="Feature Correlations",
                ),
                row=2,
                col=2,
            )

            fig_residuals.update_layout(
                title="Residuals Analysis", showlegend=False, height=1000
            )

            fig_residuals.write_html(
                str(config.forecast_vs_actual_plot).replace(".png", "_residuals.html")
            )

        except Exception as e:
            logger.error(f"Error creating plots: {str(e)}")
            raise

    def evaluate(self, config):
        """Run evaluation for multivariate forecasting"""
        logger.info("Running multivariate evaluation")

        try:
            # Load data and model
            y, X, model = self._load_data(config)

            # Setup CV splitter
            cv = ExpandingWindowSplitter(
                initial_window=int(
                    len(y) * 0.7
                ),  # Use 70% of data for initial training
                step_length=1,
                fh=list(range(1, 7)),  # Forecast horizon of 6 steps
            )

            # Create scoring metrics dictionary
            scoring = {
                "MAE": MeanAbsoluteError(),
                "MSE": MeanSquaredError(),
                "RMSE": MeanSquaredError(square_root=True),
                "MAPE": MeanAbsolutePercentageError(),
                "MASE": MeanAbsoluteScaledError(),
                "MSSE": MeanSquaredScaledError(),
                "MdSE": MedianSquaredError(),
                "MdAE": MedianAbsoluteError(),
            }

            # Run evaluation
            results = evaluate(
                forecaster=model, y=y, X=X, cv=cv, scoring=scoring, return_data=True
            )

            # Extract metrics
            metrics = {
                metric: results.loc[:, f"test_{metric}"].mean()
                for metric in scoring.keys()
                if metric in config.chosen_metrics
            }

            # Save metrics
            save_json(Path(config.scores), metrics)

            # Create and save plots
            self._create_plots(results, X, config)

            return metrics

        except Exception as e:
            logger.error(f"Error in multivariate evaluation: {str(e)}")
            raise


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

        # Check if X_test exists to determine if multivariate
        X_test_path = Path(config.test_data_dir) / "X.csv"
        self.strategy = (
            UnivariateEvaluationStrategy()
            if not X_test_path.exists()
            else MultivariateEvaluationStrategy()
        )

    def evaluate(self):
        """Run model evaluation using appropriate strategy"""
        return self.strategy.evaluate(self.config)
