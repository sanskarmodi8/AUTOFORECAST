import itertools
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import Permute, TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.impute import Imputer
from sktime.utils.plotting import plot_series

from AUTOFORECAST import logger
from AUTOFORECAST.constants import AVAIL_MODELS_GRID, AVAIL_TRANSFORMERS_GRID, DATA_DIR
from AUTOFORECAST.entity.config_entity import PreprocessingAndTrainingConfig
from AUTOFORECAST.utils.common import save_bin, save_json


class PreprocessingAndTrainingStrategy(ABC):
    @abstractmethod
    def run(self, y, X, config):
        pass


class UnivariateStrategy(PreprocessingAndTrainingStrategy):
    def _validate_and_prepare_data(self, y):
        """Validate and prepare the time series data."""
        if y is None or len(y) == 0:
            raise ValueError("Input time series is empty or None")

        # Remove any infinite values
        y = y.replace([np.inf, -np.inf], np.nan)

        # Get first and last valid index
        first_valid = y.first_valid_index()
        last_valid = y.last_valid_index()

        if first_valid is None or last_valid is None:
            raise ValueError("Time series contains no valid data")

        # Trim series to valid range
        y = y.loc[first_valid:last_valid]

        return y

    def _handle_frequency(self, y):
        """Handle time series frequency detection and resampling."""
        if len(y) <= 1:
            return y, "D"

        freq = pd.infer_freq(y.index)
        if freq is None:
            for test_freq in ["D", "B", "W", "M", "MS"]:
                resampled = y.resample(test_freq).mean()
                if len(resampled.dropna()) > len(y) * 0.8:
                    freq = test_freq
                    break
            if freq is None:
                freq = "D"

        if freq in ["MS", "M"]:
            y.index = y.index.to_period("M").to_timestamp("M")
            freq = "M"

        return y, freq

    def run(self, y, X, config):
        logger.info("Running univariate strategy")

        try:
            # Validate and prepare data
            y = self._validate_and_prepare_data(y)

            # Handle frequency
            y, freq = self._handle_frequency(y)
            logger.info(f"Using frequency: {freq}")

            # Create complete index and handle missing values
            full_idx = pd.date_range(start=y.index.min(), end=y.index.max(), freq=freq)
            y = y.reindex(full_idx)

            # Multiple imputation strategies
            y = y.interpolate(method="time", limit_direction="both")
            y = y.ffill().bfill()

            if y.isnull().values.any():
                raise ValueError(
                    "Unable to handle all missing values in the time series"
                )

            # Convert index for splitting
            y.index = pd.PeriodIndex(y.index, freq=freq)

            # Split data with validation
            if len(y) < 5:
                raise ValueError(
                    "Not enough data points for training (minimum 5 required)"
                )

            test_size = int(len(y) * 0.2)  # Convert to integer
            y_train, y_test = temporal_train_test_split(y, test_size=test_size)

            if len(y_train) < 3:
                raise ValueError("Not enough training data points after splitting")

            # Save test data
            test_data_path = Path(config.test_data_dir) / Path("y.csv")
            y_test.to_csv(test_data_path)

            # Set up cross-validation
            fh_int = np.arange(1, len(y_test) + 1)  # Integer-based forecast horizon
            initial_window = int(len(y_train) * 0.5)  # Convert to integer
            cv = ExpandingWindowSplitter(
                initial_window=initial_window, step_length=1, fh=fh_int
            )

            best_model = None
            best_params = None
            best_score = float("inf")

            for model in config.chosen_models:
                try:
                    logger.info(f"Training model: {model}")
                    param_grid = {}
                    steps = []

                    seasonal_period = {"D": 7, "W": 52, "M": 12, "Y": 1}.get(freq, 1)

                    # Add transformers
                    for transformer in config.chosen_transformers:
                        if transformer in AVAIL_TRANSFORMERS_GRID:
                            step = self._create_transformer_step(transformer)
                            if step:
                                steps.append(step)
                                param_grid.update(AVAIL_TRANSFORMERS_GRID[transformer])

                    # Add forecaster
                    forecaster_step = self._create_forecaster_step(
                        model, seasonal_period
                    )
                    if forecaster_step:
                        steps.append(forecaster_step)
                        param_grid.update(AVAIL_MODELS_GRID[model])

                    # Generate permutations
                    forecaster_name = steps[-1][0]
                    transformers_names = [step[0] for step in steps[:-1]]
                    permutations = list(itertools.permutations(transformers_names))
                    permutations = [
                        list(perm) + [forecaster_name] for perm in permutations
                    ]

                    if not permutations:
                        permutations = [[forecaster_name]]

                    param_grid = {"permutation": permutations, **param_grid}

                    # Create pipeline
                    pipe_y = TransformedTargetForecaster(steps=steps)
                    permuted_y = Permute(estimator=pipe_y, permutation=None)

                    gscv = ForecastingGridSearchCV(
                        forecaster=permuted_y,
                        param_grid=param_grid,
                        cv=cv,
                        scoring=MeanSquaredError(square_root=True),
                        n_jobs=1,  # Ensure stable behavior
                    )

                    with np.errstate(divide="ignore", invalid="ignore"):
                        gscv.fit(y=y_train)

                    if gscv.best_score_ < best_score:
                        best_model = gscv
                        best_params = {
                            "best_forecaster": model,
                            "best_params": gscv.best_params_,
                        }
                        best_score = gscv.best_score_

                except Exception as e:
                    logger.warning(f"Error fitting {model}: {str(e)}")
                    continue

            if best_model is None:
                raise ValueError("No models were successfully trained")

            # Save results
            save_bin(best_model, Path(config.model))
            save_json(Path(config.best_params), best_params)
            # evaluate the best model
            mape = MeanAbsolutePercentageError()
            y_pred = best_model.predict(fh=fh_int)
            score = mape(y_test, y_pred)
            plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
            plt.savefig(Path("eval.png"))
            logger.info(f"score_mape: {score}")

        except Exception as e:
            logger.error(f"Error in univariate strategy: {str(e)}")
            raise e

    def _create_transformer_step(self, transformer):
        """Create a transformer step based on transformer type."""
        if transformer == "Detrender":
            return ("detrender", OptionalPassthrough(Detrender()))
        elif transformer == "LogTransformer":
            return ("logtransformer", OptionalPassthrough(LogTransformer()))
        elif transformer == "ExponentTransformer":
            return ("exponenttransformer", OptionalPassthrough(ExponentTransformer()))
        elif transformer == "Imputer":
            return ("imputer", OptionalPassthrough(Imputer()))
        elif transformer == "Deseasonalizer":
            return ("deseasonalizer", OptionalPassthrough(Deseasonalizer()))
        elif transformer == "PowerTransformer":
            return (
                "powertransformer",
                OptionalPassthrough(TabularToSeriesAdaptor(PowerTransformer())),
            )
        elif transformer == "RobustScaler":
            return (
                "scaler",
                OptionalPassthrough(TabularToSeriesAdaptor(RobustScaler())),
            )
        return None

    def _create_forecaster_step(self, model, seasonal_period):
        """Create a forecaster step based on model type."""
        if model == "SARIMAX":
            return (
                "forecaster",
                SARIMAX(enforce_invertibility=False, enforce_stationarity=False),
            )
        elif model == "PolynomialTrendForecaster":
            return ("forecaster", PolynomialTrendForecaster())
        elif model == "ExponentialSmoothing":
            return ("forecaster", ExponentialSmoothing(sp=seasonal_period))
        elif model == "ThetaForecaster":
            return ("forecaster", ThetaForecaster(sp=seasonal_period))
        elif model == "NaiveForecaster":
            return ("forecaster", NaiveForecaster(sp=seasonal_period))
        elif model == "AutoARIMA":
            return (
                "forecaster",
                AutoARIMA(
                    sp=seasonal_period,
                    enforce_invertibility=False,
                    enforce_stationarity=False,
                ),
            )
        elif model == "Prophet":
            return ("forecaster", Prophet())
        elif model == "ARIMA":
            return (
                "forecaster",
                ARIMA(enforce_invertibility=False, enforce_stationarity=False),
            )
        return None


class MultivariateStrategy(PreprocessingAndTrainingStrategy):
    def _validate_and_prepare_data(self, y, X):
        """Validate and prepare both target and feature data."""
        if y is None or len(y) == 0:
            raise ValueError("Input target series is empty or None")
        if X is None or len(X) == 0:
            raise ValueError("Input feature data is empty or None")

        # Remove any infinite values
        y = y.replace([np.inf, -np.inf], np.nan)
        X = X.replace([np.inf, -np.inf], np.nan)

        # Ensure indexes match
        common_idx = y.index.intersection(X.index)
        if len(common_idx) == 0:
            raise ValueError("No common timestamps between target and features")

        y = y.loc[common_idx]
        X = X.loc[common_idx]

        # Get first and last valid index for both
        y_first = y.first_valid_index()
        y_last = y.last_valid_index()
        X_first = X.first_valid_index()
        X_last = X.last_valid_index()

        if any(idx is None for idx in [y_first, y_last, X_first, X_last]):
            raise ValueError("Target or feature data contains no valid data")

        # Get common valid range
        start_idx = max(y_first, X_first)
        end_idx = min(y_last, X_last)

        # Trim series to valid range
        y = y.loc[start_idx:end_idx]
        X = X.loc[start_idx:end_idx]

        return y, X

    def _handle_frequency(self, y, X):
        """Handle time series frequency detection and resampling for both series."""
        if len(y) <= 1:
            return y, X, "D"

        freq = pd.infer_freq(y.index)
        if freq is None:
            for test_freq in ["D", "B", "W", "M", "MS"]:
                y_resampled = y.resample(test_freq).mean()
                X_resampled = X.resample(test_freq).mean()
                if (
                    len(y_resampled.dropna()) > len(y) * 0.8
                    and len(X_resampled.dropna()) > len(X) * 0.8
                ):
                    freq = test_freq
                    break
            if freq is None:
                freq = "D"

        if freq in ["MS", "M"]:
            y.index = y.index.to_period("M").to_timestamp("M")
            X.index = X.index.to_period("M").to_timestamp("M")
            freq = "M"

        return y, X, freq

    def run(self, y, X, config):
        logger.info("Running multivariate strategy")

        try:
            # Validate and prepare data
            y, X = self._validate_and_prepare_data(y, X)

            # Handle frequency
            y, X, freq = self._handle_frequency(y, X)
            logger.info(f"Using frequency: {freq}")

            # Create complete index and handle missing values
            full_idx = pd.date_range(start=y.index.min(), end=y.index.max(), freq=freq)
            y = y.reindex(full_idx)
            X = X.reindex(full_idx)

            # Multiple imputation strategies
            y = y.interpolate(method="time", limit_direction="both")
            y = y.ffill().bfill()
            X = X.interpolate(method="time", limit_direction="both")
            X = X.ffill().bfill()

            if y.isnull().values.any() or X.isnull().values.any():
                raise ValueError("Unable to handle all missing values in the data")

            # Convert to period index
            y.index = pd.PeriodIndex(y.index, freq=freq)
            X.index = pd.PeriodIndex(X.index, freq=freq)

            # Split data with validation
            if len(y) < 5:
                raise ValueError(
                    "Not enough data points for training (minimum 5 required)"
                )

            test_size = int(len(y) * 0.2)
            y_train, y_test, X_train, X_test = temporal_train_test_split(
                y, X, test_size=test_size
            )

            if len(y_train) < 3:
                raise ValueError("Not enough training data points after splitting")

            # Save test data
            test_data_path = Path(config.test_data_dir)
            y_test.to_csv(test_data_path / "y.csv")
            X_test.to_csv(test_data_path / "X.csv")

            # Set up cross-validation
            fh = ForecastingHorizon(np.arange(1, len(y_test) + 1), is_relative=True)
            initial_window = int(len(y_train) * 0.5)
            cv = ExpandingWindowSplitter(
                initial_window=initial_window, step_length=1, fh=fh
            )

            best_model = None
            best_params = None
            best_score = float("inf")

            for model in config.chosen_models:
                try:
                    logger.info(f"Training model: {model}")
                    seasonal_period = {"D": 7, "W": 52, "M": 12, "Y": 1}.get(freq, 1)

                    # Create base parameter grid
                    param_grid = {}

                    # Create target pipeline
                    pipe_y_steps = []
                    for transformer in config.chosen_transformers:
                        if transformer in AVAIL_TRANSFORMERS_GRID:
                            step = self._create_transformer_step(transformer)
                            if step:
                                pipe_y_steps.append(step)
                                param_grid.update(
                                    {
                                        "estimator__forecaster__" + key: value
                                        for key, value in AVAIL_TRANSFORMERS_GRID[
                                            transformer
                                        ].items()
                                    }
                                )

                    # Add forecaster to target pipeline
                    forecaster_step = self._create_forecaster_step(
                        model, seasonal_period
                    )
                    if forecaster_step:
                        pipe_y_steps.append(forecaster_step)
                        param_grid.update(
                            {
                                "estimator__forecaster__" + key: value
                                for key, value in AVAIL_MODELS_GRID[model].items()
                            }
                        )

                    pipe_y = TransformedTargetForecaster(steps=pipe_y_steps)
                    permuted_y = Permute(pipe_y, permutation=None)

                    # Create exogenous pipeline
                    pipe_X_steps = []
                    for transformer in config.chosen_transformers:
                        if transformer in AVAIL_TRANSFORMERS_GRID:
                            step = self._create_transformer_step(transformer)
                            if step:
                                pipe_X_steps.append(step)
                                param_grid.update(AVAIL_TRANSFORMERS_GRID[transformer])

                    # Add permuted target pipeline as final step
                    pipe_X_steps.append(("forecaster", permuted_y))

                    pipe_X = TransformedTargetForecaster(steps=pipe_X_steps)
                    permuted_X = Permute(pipe_X, permutation=None)

                    # Generate permutations for both pipelines
                    y_transformers = [step[0] for step in pipe_y_steps[:-1]]
                    X_transformers = [step[0] for step in pipe_X_steps[:-1]]

                    y_permutations = list(
                        itertools.permutations(y_transformers + ["forecaster"])
                    )
                    X_permutations = list(
                        itertools.permutations(X_transformers + ["forecaster"])
                    )

                    if not y_permutations:
                        y_permutations = [["forecaster"]]
                    if not X_permutations:
                        X_permutations = [["forecaster"]]

                    param_grid.update(
                        {
                            "estimator__forecaster__permutation": y_permutations,
                            "permutation": X_permutations,
                        }
                    )

                    # Create and fit grid search
                    gscv = ForecastingGridSearchCV(
                        forecaster=permuted_X,
                        param_grid=param_grid,
                        cv=cv,
                        scoring=MeanSquaredError(square_root=True),
                        n_jobs=1,
                        error_score=np.nan,
                    )

                    with np.errstate(divide="ignore", invalid="ignore"):
                        gscv.fit(y=y_train, X=X_train, fh=fh)

                    if gscv.best_score_ < best_score:
                        best_model = gscv.best_forecaster_
                        best_params = {
                            "best_forecaster": model,
                            "best_params": gscv.best_params_,
                        }
                        best_score = gscv.best_score_

                except Exception as e:
                    logger.warning(f"Error fitting {model}: {str(e)}")
                    continue

            # Save results
            save_bin(best_model, Path(config.model))
            save_json(Path(config.best_params), best_params)

        except Exception as e:
            logger.error(f"Error in multivariate strategy: {str(e)}")
            raise e

    def _create_transformer_step(self, transformer):
        """Create a transformer step based on transformer type."""
        if transformer == "Detrender":
            return ("detrender", OptionalPassthrough(Detrender()))
        elif transformer == "LogTransformer":
            return ("logtransformer", OptionalPassthrough(LogTransformer()))
        elif transformer == "ExponentTransformer":
            return ("exponenttransformer", OptionalPassthrough(ExponentTransformer()))
        elif transformer == "Imputer":
            return ("imputer", OptionalPassthrough(Imputer()))
        elif transformer == "Deseasonalizer":
            return ("deseasonalizer", OptionalPassthrough(Deseasonalizer()))
        elif transformer == "PowerTransformer":
            return (
                "powertransformer",
                OptionalPassthrough(TabularToSeriesAdaptor(PowerTransformer())),
            )
        elif transformer == "RobustScaler":
            return (
                "scaler",
                OptionalPassthrough(TabularToSeriesAdaptor(RobustScaler())),
            )
        return None

    def _create_forecaster_step(self, model, seasonal_period):
        """Create a forecaster step based on model type."""
        if model == "SARIMAX":
            return (
                "forecaster",
                SARIMAX(enforce_invertibility=False, enforce_stationarity=False),
            )
        elif model == "PolynomialTrendForecaster":
            return ("forecaster", PolynomialTrendForecaster())
        elif model == "ExponentialSmoothing":
            return ("forecaster", ExponentialSmoothing(sp=seasonal_period))
        elif model == "ThetaForecaster":
            return ("forecaster", ThetaForecaster(sp=seasonal_period))
        elif model == "NaiveForecaster":
            return ("forecaster", NaiveForecaster(sp=seasonal_period))
        elif model == "AutoARIMA":
            return (
                "forecaster",
                AutoARIMA(
                    sp=seasonal_period,
                    enforce_invertibility=False,
                    enforce_stationarity=False,
                ),
            )
        elif model == "Prophet":
            return ("forecaster", Prophet())
        elif model == "ARIMA":
            return (
                "forecaster",
                ARIMA(enforce_invertibility=False, enforce_stationarity=False),
            )
        return None


class PreprocessingAndTraining:
    def __init__(self, config: PreprocessingAndTrainingConfig):
        self.config = config
        try:
            self.y = pd.read_csv(
                DATA_DIR / Path("y.csv"), index_col=0, parse_dates=True
            )
            if len(self.y) == 0:
                raise ValueError("Empty target variable data")
        except Exception as e:
            raise ValueError(f"Error reading target variable data: {str(e)}")

        try:
            self.X = pd.read_csv(
                DATA_DIR / Path("X.csv"), index_col=0, parse_dates=True
            )
        except Exception:
            self.X = None

        self.strategy = (
            UnivariateStrategy() if self.X is None else MultivariateStrategy()
        )

    def run(self):
        self.strategy.run(self.y, self.X, self.config)
