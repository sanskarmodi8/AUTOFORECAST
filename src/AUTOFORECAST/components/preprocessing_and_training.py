import itertools
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import Permute, TransformedTargetForecaster
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
    temporal_train_test_split,
)
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.boxcox import BoxCoxTransformer, LogTransformer
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.exponent import ExponentTransformer

from AUTOFORECAST import logger
from AUTOFORECAST.constants import AVAIL_MODELS_GRID, AVAIL_TRANSFORMERS_GRID, DATA_DIR
from AUTOFORECAST.entity.config_entity import PreprocessingAndTrainingConfig
from AUTOFORECAST.utils.common import save_bin, save_json


class PreprocessingAndTrainingStrategy(ABC):
    @abstractmethod
    def run(self, y, X, config):
        pass


class UnivariateWithoutExogData(PreprocessingAndTrainingStrategy):
    def _validate_and_prepare_data(self, y):
        """
        Validate and prepare the time series data.

        This function will remove any infinite values, trim the series to the first and
        last valid index, and raise an error if the time series contains no valid data.

        Parameters
        ----------
        y: pandas.Series
            The time series data to validate and prepare.

        Returns
        -------
        y: pandas.Series
            The validated and prepared time series data.
        """

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

    def _get_frequency(self, y):
        """
        Gets frequency of the time series data.

        This function will infer the frequency of the time series and convert it to a
        standard format. If the series is too short, the frequency will be set to daily.
        If the inferred frequency is MS (month start), it will be converted to M (month end).

        Parameters
        ----------
        y: pandas.Series
            The time series data to handle.

        Returns
        -------
        y: pandas.Series
            The time series data with the handled frequency.
        freq: str
            The handled frequency of the time series data.
        """

        # set freq to daily if series is too short
        if len(y) <= 1:
            return y, "D"

        # infer frequency
        freq = pd.infer_freq(y.index)

        # if freq is MS, convert to monthly (as MS is not supported as period frequency)
        if freq == "MS":
            y.index = y.index.to_period("M").to_timestamp("M")
            freq = "M"

        return y, freq

    def run(self, y, X, config):
        """
        Runs the univariate strategy.

        This function will prepare the time series data, handle any missing values,
        split the data into training and test sets, set up cross-validation,
        train and evaluate all chosen models, and save the best model.

        Parameters
        ----------
        y: pandas.Series or pandas.DataFrame
            The time series data to handle.

        X: pandas.DataFrame
            The exogenous variables data to handle.

        config: PreprocessingAndTrainingConfig
            The configuration for the univariate strategy.

        Returns
        -------
        None
        """

        logger.info("Running univariate strategy")

        try:
            # Validate and prepare data
            logger.info("Preparing data")
            y = self._validate_and_prepare_data(y)

            # Handle frequency
            y, freq = self._get_frequency(y)
            logger.info(f"Using frequency: {freq}")

            # Create final index
            full_idx = pd.date_range(start=y.index.min(), end=y.index.max(), freq=freq)
            y = y.reindex(full_idx)
            y.index = pd.PeriodIndex(y.index, freq=freq)

            # handle missing values
            y = y.interpolate(method="time", limit_direction="both")
            y = y.ffill().bfill()
            if y.isnull().values.any():
                raise ValueError(
                    "Unable to handle all missing values in the time series"
                )

            # Split data with validation
            if len(y) < 5:
                raise ValueError(
                    "Not enough data points for training (minimum 5 required)"
                )

            test_size = int(len(y) * 0.2)
            y_train, y_test = temporal_train_test_split(y, test_size=test_size)

            if len(y_train) < 3:
                raise ValueError("Not enough training data points after splitting")

            # Save test data and train data
            train_data_path = Path(config.train_data_dir) / Path("y.csv")
            y_train.to_csv(train_data_path)
            test_data_path = Path(config.test_data_dir) / Path("y.csv")
            y_test.to_csv(test_data_path)
            # Set up cross-validation
            fh = np.arange(1, len(y_test) + 1)
            initial_window = int(len(y_train) * 0.5)
            cv = ExpandingWindowSplitter(
                initial_window=initial_window, step_length=1, fh=fh
            )

            # for each chosen model, train and evaluate, save the best model
            best_model = None
            best_params = None
            best_score = float("inf")
            for model in config.chosen_models:
                try:
                    logger.info(f"Training model: {model}")
                    param_grid = {}
                    steps = []

                    # Add transformers to steps
                    for transformer in config.chosen_transformers:
                        step = self._create_transformer_step(transformer)
                        if step:
                            steps.append(step)
                            # Update parameter grid with transformer parameters
                            param_grid.update(AVAIL_TRANSFORMERS_GRID[transformer])

                    # Add forecaster to steps
                    forecaster_step = self._create_forecaster_step(model, freq)
                    if forecaster_step:
                        steps.append(forecaster_step)
                        # Update parameter grid with model parameters
                        param_grid.update(AVAIL_MODELS_GRID[model])

                    # Generate permutations of steps with forecaster as last step each time

                    transformers_names = [step[0] for step in steps[:-1]]
                    permutations = list(itertools.permutations(transformers_names))
                    permutations = [
                        list(perm) + ["forecaster"] for perm in permutations
                    ]
                    # Add permutations to parameter grid
                    param_grid = {"permutation": permutations, **param_grid}

                    # Create pipeline
                    pipe_y = TransformedTargetForecaster(steps=steps)
                    permuted_y = Permute(estimator=pipe_y)

                    # Create and fit grid search
                    gscv = ForecastingGridSearchCV(
                        forecaster=permuted_y,
                        param_grid=param_grid,
                        cv=cv,
                        error_score="raise",
                        scoring=MeanAbsolutePercentageError(symmetric=True),
                    )
                    gscv.fit(y=y_train)

                    # Update best model if current model has better score
                    logger.info(f"Best score: {best_score}")
                    logger.info(f"{model} score: {gscv.best_score_}")
                    if gscv.best_score_ < best_score:
                        best_model = gscv.best_forecaster_
                        best_params = {
                            "best_forecaster": model,
                            "best_params": gscv.best_params_,
                        }
                        best_score = gscv.best_score_

                except Exception as e:
                    logger.error(f"Error fitting {model}: {str(e)}")
                    raise e

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
        elif transformer == "BoxCoxTransformer":
            return ("boxcoxtransformer", OptionalPassthrough(BoxCoxTransformer()))
        return None

    def _create_forecaster_step(self, model, freq):
        """Create a forecaster step based on model type."""
        if model == "SARIMAX":
            return (
                "forecaster",
                SARIMAX(),
            )
        elif model == "PolynomialTrendForecaster":
            return ("forecaster", PolynomialTrendForecaster())
        elif model == "AutoARIMA":
            return (
                "forecaster",
                AutoARIMA(),
            )
        elif model == "Prophet":
            return ("forecaster", Prophet(freq=freq))
        return None


class PreprocessingAndTraining:
    def __init__(self, config: PreprocessingAndTrainingConfig):
        """
        Initialize the PreprocessingAndTraining class.

        Args:
            config (PreprocessingAndTrainingConfig): Configuration object containing
                necessary paths and parameters for preprocessing and training.

        Attributes:
            y (pd.DataFrame): Target variable data loaded from 'y.csv'.
            X (pd.DataFrame or None): Feature data loaded from 'X.csv'. If the file
                doesn't exist, X is set to None.
            strategy (PreprocessingAndTrainingStrategy): Strategy for preprocessing
                and training.

        Raises:
            ValueError: If the target variable data is empty or cannot be read.
        """

        self.config = config
        try:
            # Load target variable data
            self.y = pd.read_csv(
                DATA_DIR / Path("y.csv"), index_col=0, parse_dates=True
            )
            if len(self.y) == 0:
                raise ValueError("Empty target variable data")
        except Exception as e:
            raise ValueError(f"Error reading target variable data: {str(e)}")

        try:
            # Load feature data if available
            self.X = pd.read_csv(
                DATA_DIR / Path("X.csv"), index_col=0, parse_dates=True
            )
        except Exception:
            # If feature data doesn't exist, set X to None
            self.X = None

        # Select strategy
        if self.X is None and len(self.y.columns) == 1:
            self.strategy = UnivariateWithoutExogData()
        # TODO: Add support for more Training strategies - MultivariateWithExogData, MultivariateWithoutExogData and UnivariateWithExogData

    def run(self):
        """Run the preprocessing and training strategy."""
        self.strategy.run(self.y, self.X, self.config)
