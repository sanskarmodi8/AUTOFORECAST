from pathlib import Path

DATA_DIR = Path("artifacts/data")
AVAIL_TRANSFORMERS = [
    "Detrender",
    "LogTransformer",
    "ExponentTransformer",
    "Deseasonalizer",
    "PowerTransformer",
    "MinMaxScaler",
]
AVAIL_MODELS = [
    "NaiveForecaster",
    "AutoARIMA",
    "ExponentialSmoothing",
    "Prophet",
    "ThetaForecaster",
    "PolynomialTrendForecaster",
    "ARIMA",
    "SARIMAX",
]
AVAIL_METRICS = [
    "Mean Absolute Error",
    "Root Mean Squared Error",
    "Symmetric Mean Absolute Percentage Error",
    "Mean Absolute Scaled Error",
    "Mean Squared Scaled Error",
    "Median Squared Error",
    "Median Absolute Error",
]
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

AVAIL_MODELS_GRID = {
    "NaiveForecaster": {
        "estimator__forecaster__strategy": ["last", "mean"],
        "estimator__forecaster__sp": [4, 7, 52, 12, 365],
    },
    "AutoARIMA": {
        "estimator__forecaster__sp": [4, 7, 52, 12, 365],
        "estimator__forecaster__stationary": [True, False],
    },
    "ExponentialSmoothing": {
        "estimator__forecaster__trend": ["add", "mul"],
        "estimator__forecaster__seasonal": ["add", "mul"],
        "estimator__forecaster__damped_trend": [True, False],
        "estimator__forecaster__sp": [4, 7, 52, 12, 365],
        "estimator__forecaster__use_boxcox": [False],
        "estimator__forecaster__remove_bias": [False],
    },
    "Prophet": {
        "estimator__forecaster__seasonality_mode": ["additive", "multiplicative"],
    },
    "ThetaForecaster": {
        "estimator__forecaster__sp": [4, 7, 52, 12, 365],
        "estimator__forecaster__deseasonalize": [True, False],
    },
    "PolynomialTrendForecaster": {
        "estimator__forecaster__degree": [1, 2, 3],
    },
    "ARIMA": {
        "estimator__forecaster__simple_differencing": [True, False],
        "estimator__forecaster__enforce_stationarity": [True, False],
        "estimator__forecaster__max_iter": [50, 100],
    },
    "SARIMAX": {
        "estimator__forecaster__trend": ["n", "c", "t", "ct"],
        "estimator__forecaster__simple_differencing": [True, False],
        "estimator__forecaster__enforce_stationarity": [True, False],
    },
}

AVAIL_TRANSFORMERS_GRID = {
    "Detrender": {
        "estimator__detrender__passthrough": [True, False],
        "estimator__detrender__model": ["additive", "multiplicative"],
    },
    "LogTransformer": {
        "estimator__logtransformer__passthrough": [True, False],
    },
    "ExponentTransformer": {
        "estimator__exponenttransformer__passthrough": [True, False],
    },
    "Deseasonalizer": {
        "estimator__deseasonalizer__passthrough": [True, False],
        "estimator__deseasonalizer__sp": [4, 7, 52, 12, 365],
    },
    "PowerTransformer": {
        "estimator__powertransformer__method": ["yeo-johnson"],
        "estimator__powertransformer__standardize": [True, False],
        "estimator__powertransformer__passthrough": [True, False],
    },
    "MinMaxScaler": {
        "estimator__scaler__passthrough": [True, False],
    },
}
