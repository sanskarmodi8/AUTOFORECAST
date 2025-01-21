from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.graphics.tsaplots import acf
from statsmodels.tsa.stattools import adfuller

from AUTOFORECAST import logger
from AUTOFORECAST.constants import DATA_DIR
from AUTOFORECAST.entity.config_entity import DataAnalysisConfig
from AUTOFORECAST.utils.common import save_json


class AnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, y, config):
        pass


class UnivariateAnalysisStrategy(AnalysisStrategy):
    def analyze(self, y, config):
        """
        Analyze the time series data.

        This function will calculate the seasonal period of the time series using the
        autocorrelation function and check for stationarity using the Augmented Dickey-Fuller
        test.

        Parameters
        ----------
        y : pandas.Series
            The time series data to analyze.
        config : DataAnalysisConfig
            Configuration object containing the path to save the analysis results.

        Saves
        -----
        summary : dict
            A dictionary containing the seasonal period and whether the time series is
            stationary.
        """
        summary = {}

        # get the seasonal period
        acf_values = acf(y, len(y) - 1)
        peaks, _ = find_peaks(acf_values, distance=2)
        sp = peaks[0] if len(peaks) > 0 else None
        summary["seasonal_period"] = str(sp) if sp is not None else "1"
        logger.info(f"Detected Seasonal Period : {sp}")

        # check for stationarity
        adf_test = adfuller(y.dropna())
        summary["is_stationary"] = str(adf_test[1] < 0.05)
        logger.info(f"Stationarity in Time Series : {adf_test[1] < 0.05}")

        # save the summary
        save_json(Path(config.data_summary), summary)


class DataAnalysis:
    def __init__(self, config: DataAnalysisConfig):
        """
        Initialize the DataAnalysis class.

        Args:
            config (DataAnalysisConfig): Configuration object containing
                necessary paths and parameters for data analysis.

        Attributes:
            y (pd.DataFrame): Target variable data loaded from 'y.csv'.
            strategy (AnalysisStrategy): Strategy for data analysis.
        """

        self.config = config
        self.y = pd.read_csv(Path(DATA_DIR, "y.csv"), index_col=0, parse_dates=True)
        if len(self.y.columns) == 1:
            self.strategy = UnivariateAnalysisStrategy()
        # TODO: add support for MultivariateAnalysisStrategy

    def analyze(self):
        # Run the analysis strategy
        self.strategy.analyze(self.y, self.config)
