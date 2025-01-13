from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from sktime.utils.plotting import plot_series

from AUTOFORECAST import logger
from AUTOFORECAST.constants import AVAIL_METRICS
from AUTOFORECAST.entity.config_entity import ModelEvaluationConfig
from AUTOFORECAST.utils.common import load_bin, save_json


class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, config: ModelEvaluationConfig):
        pass


class UnivariateEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate(self, config: ModelEvaluationConfig):
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

        # Load test data and model
        y_test = pd.read_csv(Path(config.test_data_dir) / Path("y.csv"), index_col=0)
        model = load_bin(config.model)

        # get pred
        fh = np.arange(1, len(y_test) + 1)
        y_pred = model.predict(fh)

        # Evaluate model on each of the chosen metrics
        scores = {}
        for metric in config.chosen_metrics:
            logger.info("Evaluating model on metric - {}".format(metric))
            if metric == "Mean Absolute Error":
                mae = MeanAbsoluteError()
                scores[metric] = mae(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual) / Path("Mean Absolute Error.png")
                )
            elif metric == "Root Mean Squared Error":
                mse = MeanSquaredError(square_root=True)
                scores[metric] = mse(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual)
                    / Path("Root Mean Squared Error.png")
                )
            elif metric == "Symmetric Mean Absolute Percentage Error":
                smape = MeanAbsolutePercentageError(symmetric=True)
                scores[metric] = smape(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual)
                    / Path("Symmetric Mean Absolute Percentage Error.png")
                )
            elif metric == "Mean Absolute Scaled Error":
                mase = MeanAbsoluteScaledError()
                scores[metric] = mase(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual)
                    / Path("Mean Absolute Scaled Error.png")
                )
            elif metric == "Mean Squared Scaled Error":
                msse = MeanSquaredScaledError()
                scores[metric] = msse(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual)
                    / Path("Mean Squared Scaled Error.png")
                )
            elif metric == "Median Absolute Error":
                medae = MedianAbsoluteError()
                scores[metric] = medae(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual) / Path("Median Absolute Error.png")
                )
            elif metric == "Median Squared Error":
                medse = MedianSquaredError()
                scores[metric] = medse(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual) / Path("Median Squared Error.png")
                )

        # save scores
        save_json(scores, config.scores)


class MultivariateEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate(self, config: ModelEvaluationConfig):
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

        # Load test data and model
        X_test = pd.read_csv(Path(config.test_data_dir) / Path("X.csv"), index_col=0)
        y_test = pd.read_csv(Path(config.test_data_dir) / Path("y.csv"), index_col=0)
        model = load_bin(config.model)

        # get pred
        fh = np.arange(1, len(y_test) + 1)
        y_pred = model.predict(X_test, fh)

        # Evaluate model on each of the chosen metrics
        scores = {}
        for metric in config.chosen_metrics:
            logger.info("Evaluating model on metric - {}".format(metric))
            if metric == "Mean Absolute Error":
                mae = MeanAbsoluteError()
                scores[metric] = mae(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual) / Path("Mean Absolute Error.png")
                )
            elif metric == "Root Mean Squared Error":
                mse = MeanSquaredError(square_root=True)
                scores[metric] = mse(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual)
                    / Path("Root Mean Squared Error.png")
                )
            elif metric == "Symmetric Mean Absolute Percentage Error":
                smape = MeanAbsolutePercentageError(symmetric=True)
                scores[metric] = smape(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual)
                    / Path("Symmetric Mean Absolute Percentage Error.png")
                )
            elif metric == "Mean Absolute Scaled Error":
                mase = MeanAbsoluteScaledError()
                scores[metric] = mase(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual)
                    / Path("Mean Absolute Scaled Error.png")
                )
            elif metric == "Mean Squared Scaled Error":
                msse = MeanSquaredScaledError()
                scores[metric] = msse(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual)
                    / Path("Mean Squared Scaled Error.png")
                )
            elif metric == "Median Absolute Error":
                medae = MedianAbsoluteError()
                scores[metric] = medae(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual) / Path("Median Absolute Error.png")
                )
            elif metric == "Median Squared Error":
                medse = MedianSquaredError()
                scores[metric] = medse(y_test, y_pred)
                plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
                plt.savefig(
                    Path(config.forecast_vs_actual) / Path("Median Squared Error.png")
                )

        # save scores
        save_json(scores, config.scores)


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize ModelEvaluation with appropriate strategy.

        Args:
            config (ModelEvaluationConfig): ModelEvaluationConfig object containing
                necessary paths and parameters for model evaluation.

        Attributes:
            strategy (ModelEvaluationStrategy): Strategy for model evaluation, selected
                based on whether feature data is available (univariate or multivariate).
        """
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
