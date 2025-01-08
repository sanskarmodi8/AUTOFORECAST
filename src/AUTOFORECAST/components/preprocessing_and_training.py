import itertools
from abc import ABC, abstractmethod
from pathlib import Path

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
        """
        Method to run the preprocessing and training strategy.

        Parameters:
        y (pd.Series): Target variable
        X (pd.DataFrame): Feature variables
        config (PreprocessingAndTrainingConfig): Configuration for preprocessing and training.
        """
        pass


class UnivariateStrategy(PreprocessingAndTrainingStrategy):
    def run(self, y, X, config):
        logger.info("Running univariate strategy")

        # Ensure the time series has a frequency
        logger.info("Setting time series frequency")
        # Infer frequency if possible, otherwise default to daily
        if len(y) > 1:
            freq = pd.infer_freq(y.index)
            logger.info(f"Detected frequency: {freq}")
        else:
            freq = "D"
            logger.info(
                "Not enough data points to infer frequency, defaulting to daily"
            )

        # Reindex the time series with the determined frequency
        y = y.asfreq(freq)

        logger.info("Splitting data into train and test")
        # Split data into train and test
        y_train, y_test = temporal_train_test_split(y, test_size=0.2)

        # Save test data for later evaluation
        y_test.to_csv(Path(config.test_data_dir) / Path("y.csv"))

        # Create expanding window splitter for cross-validation
        fh = ForecastingHorizon(y_test.index, is_relative=False).to_relative(
            y_train.index[-1]
        )
        cv = ExpandingWindowSplitter(
            fh=fh, initial_window=int(len(y_train) * 0.5), step_length=1
        )

        # Rest of your existing code remains the same
        best_model = None
        best_params = None
        best_score = None

        for model in config.chosen_models:
            # Build the parameter grid for tuning by adding chosen transformers and models
            logger.info(f"Building parameter grid for {model}")
            param_grid = {}
            for transformer in config.chosen_transformers:
                param_grid.update(AVAIL_TRANSFORMERS_GRID[transformer])
            param_grid.update(AVAIL_MODELS_GRID[model])

            # Add the transformers and models to the pipeline
            steps = []

            for transformer in config.chosen_transformers:
                if transformer == "Detrender":
                    steps.append(("detrender", OptionalPassthrough(Detrender())))
                if transformer == "LogTransformer":
                    steps.append(
                        ("logtransformer", OptionalPassthrough(LogTransformer()))
                    )
                if transformer == "ExponentTransformer":
                    steps.append(
                        (
                            "exponenttransformer",
                            OptionalPassthrough(ExponentTransformer()),
                        )
                    )
                if transformer == "Imputer":
                    steps.append(("imputer", OptionalPassthrough(Imputer())))
                if transformer == "Deseasonalizer":
                    steps.append(
                        ("deseasonalizer", OptionalPassthrough(Deseasonalizer()))
                    )
                if transformer == "PowerTransformer":
                    steps.append(
                        (
                            "powertransformer",
                            OptionalPassthrough(
                                TabularToSeriesAdaptor(PowerTransformer())
                            ),
                        )
                    )
                if transformer == "RobustScaler":
                    steps.append(
                        (
                            "scaler",
                            OptionalPassthrough(TabularToSeriesAdaptor(RobustScaler())),
                        )
                    )

            if model == "SARIMAX":
                steps.append(("forecaster", SARIMAX()))
            elif model == "PolynomialTrendForecaster":
                steps.append(("forecaster", PolynomialTrendForecaster()))
            elif model == "ExponentialSmoothing":
                steps.append(("forecaster", ExponentialSmoothing()))
            elif model == "ThetaForecaster":
                steps.append(("forecaster", ThetaForecaster()))
            elif model == "NaiveForecaster":
                steps.append(("forecaster", NaiveForecaster()))
            elif model == "AutoARIMA":
                steps.append(("forecaster", AutoARIMA()))
            elif model == "Prophet":
                steps.append(("forecaster", Prophet()))
            elif model == "ARIMA":
                steps.append(("forecaster", ARIMA()))

            # get permutations of steps with forecaster as the last step and update param_grid
            forecaster_name = steps[-1][0]
            transformers_names = [step[0] for step in steps[:-1]]
            permutations = list(itertools.permutations(transformers_names))
            permutations = [list(perm) + [forecaster_name] for perm in permutations]
            # add permutations to param_grid at the start
            param_grid = {"permutation": permutations, **param_grid}
            logger.info(f"Steps: {steps}")
            logger.info(f"Parameter grid: {param_grid}")
            # perform grid search cv
            pipe_y = TransformedTargetForecaster(steps=steps)
            permuted_y = Permute(estimator=pipe_y, permutation=None)

            logger.info(f"Performing grid search cv for model {model}")
            gscv = ForecastingGridSearchCV(
                forecaster=permuted_y,
                param_grid=param_grid,
                cv=cv,
                verbose=1,
                scoring=MeanSquaredError(square_root=True),
                error_score="raise",
                n_jobs=-1,
            )
            gscv.fit(y=y_train, fh=fh)

            # Store the best model
            if best_score is None or gscv.best_score_ < best_score:
                best_model = gscv.best_estimator_
                best_params = {
                    "best_forecaster": model,
                    "best_params": gscv.best_params_,
                }
                best_score = gscv.best_score_

        # Save the best model and its parameters
        save_bin(best_model, config.model)
        save_json(Path(config.best_params), best_params)


class MultivariateStrategy(PreprocessingAndTrainingStrategy):
    def run(self, y, X, config):
        pass


class PreprocessingAndTraining:
    def __init__(self, config: PreprocessingAndTrainingConfig):
        """
        Constructor for PreprocessingAndTraining class.

        This class is used to select appropriate preprocessing and training
        strategy based on whether the data is univariate or multivariate.

        Parameters:
        config (PreprocessingAndTrainingConfig): Configuration for preprocessing
            and training.
        """
        self.config = config
        self.y = pd.read_csv(DATA_DIR / Path("y.csv"), index_col=0, parse_dates=True)
        try:
            self.X = pd.read_csv(
                DATA_DIR / Path("X.csv"), index_col=0, parse_dates=True
            )
        except Exception:
            self.X = None
        if self.X is None:
            self.strategy = UnivariateStrategy()
        else:
            self.strategy = MultivariateStrategy()

    def run(self):
        """
        Method to run the preprocessing and training strategy.
        """
        self.strategy.run(self.y, self.X, self.config)
