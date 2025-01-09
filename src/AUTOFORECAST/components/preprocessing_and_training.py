import itertools
from abc import ABC, abstractmethod
from pathlib import Path

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
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.impute import Imputer

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
                        best_model = gscv.best_forecaster_
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
    def run(self, y, X, config):
        logger.info("Multivariate strategy not implemented yet")
        pass


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
