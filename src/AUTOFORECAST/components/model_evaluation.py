from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MedianAbsoluteError,
    MedianSquaredError,
)
from sktime.utils.plotting import plot_series

from AUTOFORECAST import logger
from AUTOFORECAST.constants import AVAIL_METRICS
from AUTOFORECAST.entity.config_entity import ModelEvaluationConfig
from AUTOFORECAST.utils.common import create_directories, load_bin, save_json

# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, y_train, y_test, X_test, config: ModelEvaluationConfig):
        pass


class UnivariateWithoutExogData(ModelEvaluationStrategy):
    def evaluate(self, y_train, y_test, X_test, config: ModelEvaluationConfig):
        """
        Evaluate the model using the specified metrics and save the evaluation results.

        This method loads the test data and a pre-trained model, uses the model to
        predict future values, and evaluates the prediction using the metrics specified
        in the configuration. The evaluation scores are saved to a JSON file, and plots
        comparing the predicted and actual values are saved for each metric.

        Args:
            config (ModelEvaluationConfig): Configuration object containing paths to
            the model, test data, chosen metrics, and directories for saving scores
            and plots.

        """

        # Load the model
        model = load_bin(Path(config.model))

        # get pred
        fh = np.arange(1, len(y_test) + 1)
        y_pred = model.predict(fh)
        y_pred.index = y_pred.index.to_timestamp()

        plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
        plt.savefig(BASE_DIR / Path(config.forecast_vs_actual_plot))

        # Evaluate model on each of the chosen metrics
        scores = {}
        for metric in config.chosen_metrics:
            logger.info("Evaluating model on metric - {}".format(metric))
            if metric == "Mean Absolute Error":
                mae = MeanAbsoluteError()
                scores[metric] = mae(y_test, y_pred)

            elif metric == "Root Mean Squared Error":
                mse = MeanSquaredError(square_root=True)
                scores[metric] = mse(y_test, y_pred)

            elif metric == "Symmetric Mean Absolute Percentage Error":
                smape = MeanAbsolutePercentageError(symmetric=True)
                scores[metric] = smape(y_test, y_pred)

            elif metric == "Median Absolute Error":
                medae = MedianAbsoluteError()
                scores[metric] = medae(y_test, y_pred)

            elif metric == "Median Squared Error":
                medse = MedianSquaredError()
                scores[metric] = medse(y_test, y_pred)

        # save scores
        logger.info(f"Evaluated scores : {scores}")
        save_json(BASE_DIR / Path(config.scores), scores)


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize the ModelEvaluation class.

        Args:
            config (ModelEvaluationConfig): Configuration object containing
                necessary paths and parameters for model evaluation.

        Attributes:
            y_test (pd.DataFrame): Target variable data loaded from 'y.csv'.
            x_test (pd.DataFrame or None): Feature data loaded from 'X.csv'. If the file
                doesn't exist, x_test is set to None.
            strategy (ModelEvaluationStrategy): Strategy for model evaluation

        Raises:
            ValueError: If the target variable data is empty or cannot be read.
        """
        self.config = config

        try:
            # Load target variable data
            self.y_test = pd.read_csv(
                f"{BASE_DIR / Path(config.test_data_dir)}/y.csv",
                index_col=0,
                parse_dates=True,
            )
            if len(self.y_test) == 0:
                raise ValueError("Empty target variable data")
        except Exception as e:
            raise ValueError(f"Error reading target variable data: {str(e)}")

        try:
            # load the train data for plotting
            self.y_train = pd.read_csv(
                f"{BASE_DIR / Path(config.train_data_dir)}/y.csv",
                index_col=0,
                parse_dates=True,
            )
            if len(self.y_train) == 0:
                raise ValueError("Empty train data")
        except Exception as e:
            raise ValueError(f"Error reading train data: {str(e)}")
        try:
            # load feature data if available
            self.x_test = pd.read_csv(
                f"{BASE_DIR / Path(config.test_data_dir)}/X.csv",
                index_col=0,
                parse_dates=True,
            )
        except Exception:
            # if feature data is not available, set to None
            self.x_test = None

        # Select strategy
        if self.x_test is None and len(self.y_train.columns) == 1:
            self.strategy = UnivariateWithoutExogData()
        # TODO: Add support for more Evaluation strategies - MultivariateWithExogData, MultivariateWithoutExogData and UnivariateWithExogData

    def evaluate(self):
        """Run model evaluation using appropriate strategy"""
        self.strategy.evaluate(self.y_train, self.y_test, self.x_test, self.config)
