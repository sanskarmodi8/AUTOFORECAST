import itertools
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.auto_arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import Permute, TransformedTargetForecaster
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.prophet import Prophet
from sktime.forecasting.random_forest import RandomForestForecaster
from sktime.forecasting.sarima import SARIMA
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.seasonal import SeasonalNaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series import (
    Deseasonalizer,
    Detrender,
    ExponentTransformer,
    Imputer,
    LogTransformer,
)
from sktime.transformations.series.adapt import TabularToSeriesAdaptor

from AUTOFORECAST import logger
from AUTOFORECAST.constants import AVAIL_MODELS_GRID, AVAIL_TRANSFORMERS_GRID, DATA_DIR
from AUTOFORECAST.entity.config_entity import PreprocessingAndTrainingConfig
from AUTOFORECAST.utils.common import load_json, save_bin


class PreprocessingAndTrainingStrategy(ABC):
    @abstractmethod
    def run(self, y, X, config, data_summary):
        """
        Method to run the preprocessing and training strategy.

        Parameters:
        y (pd.Series): Target variable
        X (pd.DataFrame): Feature variables
        config (PreprocessingAndTrainingConfig): Configuration for preprocessing and training.
        """
        pass


class UnivariateStrategy(PreprocessingAndTrainingStrategy):
    def run(self, y, X, config, data_summary):
        logger.info("Running univariate strategy")
        logger.info("Splitting data into train and test")

        # Split data into train and test
        y_train, y_test = temporal_train_test_split(y, test_size=0.2)

        # Save test data for later evaluation
        y_test.to_csv(config.test_data_dir / "y.csv")

        # Create expanding window splitter for cross-validation
        fh = ForecastingHorizon(y_test.index, is_relative=False).to_relative(
            y_train.index[-1]
        )
        cv = ExpandingWindowSplitter(
            fh=fh, initial_window=int(len(y_train) * 0.5), step_length=1
        )

        # Build the parameter grid for tuning by adding chosen transformers and models
        param_grid = {}
        for transformer in config.chosen_transformers:
            param_grid.update(AVAIL_TRANSFORMERS_GRID[transformer])
            if transformer == "Deseasonalizer":
                param_grid["estimator__deseasonalizer__sp"].append(
                    data_summary.seasonal_period
                )
        for model in config.chosen_models:
            param_grid.update(AVAIL_MODELS_GRID[model])

        # Add the transformers and models to the pipeline
        steps = []

        for transformer in config.chosen_transformers:
            if transformer == "Detrender":
                steps.append(("detrender", OptionalPassthrough(Detrender())))
            if transformer == "LogTransformer":
                steps.append(("logtransformer", OptionalPassthrough(LogTransformer())))
            if transformer == "ExponentTransformer":
                steps.append(
                    ("exponenttransformer", OptionalPassthrough(ExponentTransformer()))
                )
            if transformer == "Imputer":
                steps.append(("imputer", OptionalPassthrough(Imputer())))
            if transformer == "MinMaxScaler":
                steps.append(
                    (
                        "scaler",
                        OptionalPassthrough(TabularToSeriesAdaptor(MinMaxScaler())),
                    )
                )
            if transformer == "Deseasonalizer":
                steps.append(("deseasonalizer", OptionalPassthrough(Deseasonalizer())))
            if transformer == "StandardScaler":
                steps.append(("scaler", OptionalPassthrough(StandardScaler())))
            if transformer == "PowerTransformer":
                steps.append(
                    (
                        "powertransformer",
                        OptionalPassthrough(TabularToSeriesAdaptor(PowerTransformer())),
                    )
                )
            if transformer == "RobustScaler":
                steps.append(
                    (
                        "scaler",
                        OptionalPassthrough(TabularToSeriesAdaptor(RobustScaler())),
                    )
                )

        for model in config.chosen_models:
            if model == "ARIMA":
                steps.append(("forecaster", ARIMA()))
            elif model == "SARIMA":
                steps.append(("forecaster", SARIMA()))
            elif model == "SARIMAX":
                steps.append(("forecaster", SARIMAX()))
            elif model == "RandomForestForecaster":
                steps.append(("forecaster", RandomForestForecaster()))
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
            elif model == "SeasonalNaiveForecaster":
                steps.append(("forecaster", SeasonalNaiveForecaster()))

        # get permutations of steps with forecaster as the last step and update param_grid
        forecaster_name = steps[-1][0]
        transformers_names = [step[0] for step in steps[:-1]]
        permutations = list(itertools.permutations(transformers_names))
        permutations = [list(perm) + [forecaster_name] for perm in permutations]
        param_grid.update({"permutation": permutations})

        # perform grid search cv
        pipe_y = TransformedTargetForecaster(steps=steps)
        permuted_y = Permute(estimator=pipe_y, permutation=None)

        logger.info("Performing grid search cv")
        gscv = ForecastingGridSearchCV(
            forecaster=permuted_y,
            param_grid=param_grid,
            cv=cv,
            verbose=1,
            scoring=MeanSquaredError(square_root=True),
            error_score="raise",
        )
        gscv.fit(y=y_train, fh=fh)

        # Save the best model
        best_model = gscv.best_estimator_
        save_bin(best_model, config.model)
        logger.info("Saved the best model")


class MultivariateStrategy(PreprocessingAndTrainingStrategy):
    def run(self, y, X, config, data_summary):
        logger.info("Running multivariate strategy")
        logger.info("Splitting data into train and test")

        # Split the data
        y_train, y_test, X_train, X_test = temporal_train_test_split(
            y, X, test_size=0.2
        )

        # Save test data for later evaluation
        y_test.to_csv(config.test_data_dir / "y.csv")
        X_test.to_csv(config.test_data_dir / "X.csv")

        # Create expanding window splitter for cross-validation
        fh = ForecastingHorizon(y_test.index, is_relative=False).to_relative(
            y_train.index[-1]
        )
        cv = ExpandingWindowSplitter(
            fh=fh, initial_window=int(len(y_train) * 0.5), step_length=1
        )

        # Build the parameter grid for tuning by adding chosen transformers and models
        param_grid = {}
        for transformer in config.chosen_transformers:
            param_grid.update(AVAIL_TRANSFORMERS_GRID[transformer])  # for X
            param_grid.update(
                "estimator__forecaster__" + AVAIL_TRANSFORMERS_GRID[transformer]
            )  # for y
            if transformer == "Deseasonalizer":
                param_grid["estimator__deseasonalizer__sp"].append(
                    data_summary.seasonal_period
                )
        for model in config.chosen_models:
            param_grid.update(
                "estimator__forecaster__" + AVAIL_MODELS_GRID[model]
            )  # for y

        # Add the transformers and models to the pipeline
        steps = []

        for transformer in config.chosen_transformers:
            if transformer == "Detrender":
                steps.append(("detrender", OptionalPassthrough(Detrender())))
            if transformer == "LogTransformer":
                steps.append(("logtransformer", OptionalPassthrough(LogTransformer())))
            if transformer == "ExponentTransformer":
                steps.append(
                    ("exponenttransformer", OptionalPassthrough(ExponentTransformer()))
                )
            if transformer == "Imputer":
                steps.append(("imputer", OptionalPassthrough(Imputer())))
            if transformer == "Deseasonalizer":
                steps.append(("deseasonalizer", OptionalPassthrough(Deseasonalizer())))
            if transformer == "StandardScaler":
                steps.append(("standardscaler", OptionalPassthrough(StandardScaler())))
            if transformer == "MinMaxScaler":
                steps.append(("minmaxscaler", OptionalPassthrough(MinMaxScaler())))
            if transformer == "PowerTransformer":
                steps.append(
                    ("powerttransformer", OptionalPassthrough(PowerTransformer()))
                )
            if transformer == "RobustScaler":
                steps.append(("robustscaler", OptionalPassthrough(RobustScaler())))

        for model in config.chosen_models:
            if model == "ARIMA":
                steps.append(("forecaster", ARIMA()))
            elif model == "ThetaForecaster":
                steps.append(("forecaster", ThetaForecaster()))
            elif model == "NaiveForecaster":
                steps.append(("forecaster", NaiveForecaster()))
            elif model == "AutoARIMA":
                steps.append(("forecaster", AutoARIMA()))
            elif model == "Prophet":
                steps.append(("forecaster", Prophet()))
            elif model == "SeasonalNaiveForecaster":
                steps.append(("forecaster", SeasonalNaiveForecaster()))
            elif model == "PolynomialTrendForecaster":
                steps.append(("forecaster", PolynomialTrendForecaster()))
            elif model == "RandomForestForecaster":
                steps.append(("forecaster", RandomForestForecaster()))
            elif model == "SARIMA":
                steps.append(("forecaster", SARIMA()))
            elif model == "SARIMAX":
                steps.append(("forecaster", SARIMAX()))
            elif model == "ExponentialSmoothing":
                steps.append(("forecaster", ExponentialSmoothing()))

        # get permutations of steps with forecaster as the last step and update param_grid
        forecaster_name = steps[-1][0]
        transformers_names = [step[0] for step in steps[:-1]]
        permutations = list(itertools.permutations(transformers_names))
        permutations = [list(perm) + [forecaster_name] for perm in permutations]
        param_grid.update({"permutation": permutations})  # for X
        param_grid.update({"estimator__forecaster__permutation": permutations})  # for y

        # perform grid search cv
        logger.info("Performing grid search cv")
        pipe_y = TransformedTargetForecaster(steps=steps)
        permuted_y = Permute(estimator=pipe_y, permutation=None)
        steps_x = steps[:-1]
        steps_x.append(("forecaster", permuted_y))
        pipe_x = TransformedTargetForecaster(steps=steps_x)
        permuted_x = Permute(estimator=pipe_x, permutation=None)
        gscv = ForecastingGridSearchCV(
            forecaster=permuted_x,
            param_grid=param_grid,
            cv=cv,
            verbose=1,
            scoring=MeanSquaredError(square_root=True),
            error_score="raise",
        )
        gscv.fit(y=y_train, X=X_train, fh=fh)

        # Save the best model
        best_model = gscv.best_estimator_
        save_bin(best_model, config.model)
        logger.info("Saved the best model")


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
        self.y = pd.read_csv(DATA_DIR / "y.csv")
        try:
            self.X = pd.read_csv(DATA_DIR / "X.csv")
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
        data_summary = load_json(self.config.data_summary)
        self.strategy.run(self.y, self.X, self.config, data_summary)
