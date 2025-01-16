from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.utils.plotting import plot_series

from AUTOFORECAST import logger
from AUTOFORECAST.entity.config_entity import ForecastingConfig
from AUTOFORECAST.utils.common import load_bin


class Forecasting:
    def __init__(self, config: ForecastingConfig):
        self.config = config
        self.model = load_bin(Path(config.model))
        self.y_test = pd.read_csv(
            Path(config.test_data_dir) / Path("y.csv"), parse_dates=True, index_col=0
        )
        self.y_train = pd.read_csv(
            Path(config.train_data_dir) / Path("y.csv"), parse_dates=True, index_col=0
        )

    def forecast(self):

        # get pred
        fh = np.arange(len(self.y_test) + 1, len(self.y_test) + self.config.fh + 1)
        y_pred = self.model.predict(fh)
        y_pred.index = y_pred.index.to_timestamp()

        # save the forecast plot
        plot_series(
            self.y_train,
            self.y_test,
            y_pred,
            labels=["y_train", "y_test", "forecast_with_given_fh"],
        )
        plt.savefig(self.config.forecast_plot)

        # save forecast as csv
        forecast_data = pd.DataFrame(y_pred)
        forecast_data.to_csv(Path(self.config.forecast_data), index=False)
