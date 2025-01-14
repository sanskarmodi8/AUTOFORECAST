from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.utils.plotting import plot_series

from AUTOFORECAST.entity.config_entity import ForecastingConfig
from AUTOFORECAST.utils.common import load_bin


class Forecasting:
    def __init__(self, config: ForecastingConfig):
        self.config = config
        self.model = load_bin(config.model)

    def forecast(self):

        # get pred
        fh = np.arange(1, len(self.config.fh) + 1)
        y_pred = self.model.predict(fh)

        # save the forecast plot
        plot_series(y_pred, labels=["y_pred"])
        plt.savefig(self.config.forecast_plot)

        # save forecast as csv
        forecast_data = pd.DataFrame({"fh": fh, "y_pred": y_pred})
        forecast_data.to_csv(Path(self.config.forecast_data), index=False)
