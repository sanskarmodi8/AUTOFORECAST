import json
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from AUTOFORECAST import logger
from AUTOFORECAST.constants import DATA_DIR
from AUTOFORECAST.entity.config_entity import DataAnalysisConfig
from AUTOFORECAST.utils.common import save_json


class DataAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, series):
        pass


# Concrete Strategy for Seasonality Detection
class SeasonalityDetectionStrategy(DataAnalysisStrategy):
    def analyze(self, series):
        """
        Detects the seasonal period (SP) of the time series data in the given column.

        Args:
        series (pd.Series): The column containing the time series data.

        Returns:
        int: The estimated seasonal period (SP).
        """
        min_periods = 1
        max_periods = 365

        best_sp = None
        highest_corr = -np.inf

        logger.info("Detecting seasonality...")

        # Iterate over different seasonal periods (periodicity)
        for sp in range(min_periods, max_periods + 1):
            try:
                stl = STL(series, seasonal=sp, robust=True)
                result = stl.fit()
                seasonal_component = result.seasonal

                # Measure autocorrelation (correlation of the seasonal component with lagged version)
                autocorr = np.corrcoef(seasonal_component[:-1], seasonal_component[1:])[
                    0, 1
                ]

                if autocorr > highest_corr:
                    highest_corr = autocorr
                    best_sp = sp
            except Exception as e:
                logger.error(f"Error with seasonal period {sp}: {e}")

        logger.info(f"Seasonal period detected: {best_sp}")
        result = {"seasonal_period": best_sp}
        return result


# Context class for data analysis
class DataAnalysis:
    def __init__(self, config: DataAnalysisConfig):
        self.config = config
        self.strategy = SeasonalityDetectionStrategy()

    def execute_analysis(self):

        # Load the data

        df = pd.read_csv(DATA_DIR + "y.csv")
        series = df.iloc[:, 0]

        # Execute analysis and save the result
        result = self.strategy.analyze(series)
        save_json(result, self.config.data_summary)
