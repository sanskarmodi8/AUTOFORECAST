from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.utils.plotting import plot_series

from AUTOFORECAST import logger
from AUTOFORECAST.entity.config_entity import ForecastingConfig
from AUTOFORECAST.utils.common import load_bin

# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class ForecastingStrategy(ABC):
    @abstractmethod
    def forecast(self, y_test, y_train, config):
        pass


class UnivariateForecastingStrategy(ForecastingStrategy):
    def forecast(self, y_test, y_train, config):
        """
        Make forecast using the trained model, save the forecast plot and forecasted data.

        Args:
            y_test (pd.DataFrame): Target variable data loaded from 'y.csv'.
            y_train (pd.DataFrame): Data used for training the model.
            config (ForecastingConfig): Configuration object containing
                necessary paths and parameters for forecasting.

        Saves:
            forecast_plot (Path): Path to the forecast plot.
            forecast_data (Path): Path to the forecasted data.
        """

        # Load the model
        model = load_bin(Path(config.model))

        # get pred
        fh = np.arange(len(y_test) + 1, len(y_test) + config.fh + 1)
        y_pred = model.predict(fh)
        y_pred.index = y_pred.index.to_timestamp()

        # save the forecast plot
        plot_series(
            y_train,
            y_test,
            y_pred,
            labels=["y_train", "y_test", "forecast_with_given_fh"],
        )
        plt.savefig(BASE_DIR / Path(config.forecast_plot))

        # save forecast as csv
        forecast_data = pd.DataFrame(y_pred)
        forecast_data.to_csv(f"{BASE_DIR / Path(config.forecast_data)}", index=False)


class Forecasting:
    def __init__(self, config: ForecastingConfig):
        """
        Initialize the Forecasting class.

        Args:
            config (ForecastingConfig): Configuration object containing
                necessary paths and parameters for forecasting.

        Attributes:
            y_test (pd.DataFrame): Target variable data loaded from 'y.csv'.
            y_train (pd.DataFrame): Data used for training the model.
            strategy (ForecastingStrategy): Strategy for forecasting.

        """
        self.config = config
        self.y_test = pd.read_csv(
            f"{BASE_DIR / Path(config.test_data_dir)}/y.csv",
            parse_dates=True,
            index_col=0,
        )
        self.y_train = pd.read_csv(
            f"{BASE_DIR / Path(config.train_data_dir)}/y.csv",
            parse_dates=True,
            index_col=0,
        )

        if len(self.y_train.columns) == 1:
            self.strategy = UnivariateForecastingStrategy()
        # TODO: Add support for MultivariateForecastingStrategy

    def forecast(self):
        # use the appropriate strategy to forecast
        self.strategy.forecast(self.y_test, self.y_train, self.config)
