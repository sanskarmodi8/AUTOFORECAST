import itertools
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import Permute, TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster, STLForecaster
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.transformations.compose import OptionalPassthrough, TransformerPipeline
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.boxcox import BoxCoxTransformer, LogTransformer
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.impute import Imputer

from AUTOFORECAST import logger
from AUTOFORECAST.constants import AVAIL_MODELS_GRID, AVAIL_TRANSFORMERS_GRID, DATA_DIR
from AUTOFORECAST.entity.config_entity import PreprocessingAndTrainingConfig
from AUTOFORECAST.utils.common import load_json, save_bin, save_json

# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


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

            # get sp and check for stationarity
            sp = int(load_json(Path(config.data_summary)).seasonal_period)
            is_stationary = (
                False
                if load_json(Path(config.data_summary)).is_stationary == "False"
                else True
            )

            # Validate and prepare data
            logger.info("Preparing data")
            y = self._validate_and_prepare_data(y)

            # Handle frequency
            y, freq = self._get_frequency(y)
            logger.info(f"Using frequency: {freq}")

            # Create final index
            full_idx = pd.date_range(start=y.index.min(), end=y.index.max(), freq=freq)
            y = y.reindex(full_idx)

            # if freq inferred was None, reinfer after reindexing
            if freq is None:
                freq = self._get_frequency(y)[1]
                logger.info(f"Reinferred frequency: {freq}")

            # Convert to period index
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
            create_directories(
                [
                    BASE_DIR / Path(config.train_data_dir),
                    BASE_DIR / Path(config.test_data_dir),
                ]
            )
            y_train.to_csv(f"{BASE_DIR / Path(config.train_data_dir)}/y.csv")
            y_test.to_csv(f"{BASE_DIR / Path(config.test_data_dir)}/y.csv")

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
                        step = self._create_transformer_step(transformer, sp)
                        if step:
                            steps.append(step)
                            # Update parameter grid with transformer parameters
                            param_grid.update(AVAIL_TRANSFORMERS_GRID[transformer])

                            # cautiously set detrender model to avoid negative values
                            if transformer == "Detrender":
                                y_transformed_add = Detrender(
                                    model="additive"
                                ).fit_transform(y_train)
                                y_transformed_mul = Detrender(
                                    model="multiplicative"
                                ).fit_transform(y_train)

                                if (y_transformed_add < 0).any().any() and not (
                                    y_transformed_mul < 0
                                ).any().any():
                                    param_grid.update(
                                        {
                                            "estimator__detrender__model": [
                                                "multiplicative"
                                            ]
                                        }
                                    )
                                elif (y_transformed_mul < 0).any().any() and not (
                                    y_transformed_add < 0
                                ).any().any():
                                    param_grid.update(
                                        {"estimator__detrender__model": ["additive"]}
                                    )
                                elif (y_transformed_add < 0).any().any() and (
                                    y_transformed_mul < 0
                                ).any().any():
                                    # remove detrender from steps and param_grid
                                    steps = [
                                        step for step in steps if step[0] != "detrender"
                                    ]
                                    param_grid = {
                                        key: value
                                        for key, value in param_grid.items()
                                        if "detrender" not in key
                                    }
                                else:
                                    param_grid.update(
                                        {
                                            "estimator__detrender__model": [
                                                "additive",
                                                "multiplicative",
                                            ]
                                        }
                                    )

                    # Update parameter grid with imputer parameters
                    param_grid.update(
                        {
                            f"estimator__{step[0]}__imputer__method": [
                                "mean",
                                "drift",
                                "nearest",
                            ]
                            for step in steps
                            if step[0] != "forecaster"
                        }
                    )

                    # Add forecaster to steps
                    forecaster_step = self._create_forecaster_step(
                        model, freq, sp, is_stationary
                    )
                    if forecaster_step:
                        steps.append(forecaster_step)
                        # Update parameter grid with model parameters
                        param_grid.update(AVAIL_MODELS_GRID[model])

                    logger.info(f"Steps: {steps}")
                    logger.info(f"Parameter grid: {param_grid}")

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
            logger.info(f"Best model: {best_model}")
            logger.info(f"Best params: {best_params}")
            save_bin(best_model, BASE_DIR / Path(config.model))
            save_json(BASE_DIR / Path(config.best_params), best_params)

        except Exception as e:
            logger.error(f"Error in univariate strategy: {str(e)}")
            raise e

    def _create_transformer_step(self, transformer, sp):
        """Create a transformer step based on transformer type."""
        step = None
        imputer = Imputer()
        if transformer == "ExponentTransformer":
            step = ("exponenttransformer", OptionalPassthrough(ExponentTransformer()))
        elif transformer == "LogTransformer":
            step = ("logtransformer", OptionalPassthrough(LogTransformer()))
        elif transformer == "Deseasonalizer":
            step = ("deseasonalizer", OptionalPassthrough(Deseasonalizer(sp=sp)))
        elif transformer == "BoxCoxTransformer":
            step = ("boxcoxtransformer", OptionalPassthrough(BoxCoxTransformer()))

        elif transformer == "Detrender":
            step = ("detrender", OptionalPassthrough(Detrender()))

        if step:
            return (
                step[0],
                TransformerPipeline(
                    steps=[
                        (transformer, step[1]),
                        ("imputer", imputer),
                    ]
                ),
            )
        return None

    def _create_forecaster_step(self, model, freq, sp, is_stationary):
        """Create a forecaster step based on model type."""
        if model == "PolynomialTrendForecaster":
            return ("forecaster", PolynomialTrendForecaster())
        elif model == "Prophet":
            return ("forecaster", Prophet(freq=freq))
        elif model == "AutoARIMA":
            return ("forecaster", AutoARIMA(sp=sp, stationary=is_stationary))
        elif model == "ExponentialSmoothing":
            if sp == 1:
                return ("forecaster", ExponentialSmoothing())
            return ("forecaster", ExponentialSmoothing(sp=sp))
        elif model == "ThetaForecaster":
            return ("forecaster", ThetaForecaster(sp=sp))
        elif model == "STLForecaster":
            if sp == 1:
                return ("forecaster", STLForecaster())
            return ("forecaster", STLForecaster(sp=sp))
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
