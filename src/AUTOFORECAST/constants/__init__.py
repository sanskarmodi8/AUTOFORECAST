from pathlib import Path

DATA_DIR = Path("artifacts/data")
AVAIL_TRANSFORMERS = [
    "Detrender",
    "LogTransformer",
    "ExponentTransformer",
    "BoxCoxTransformer",
]
AVAIL_MODELS = [
    "Prophet",
    "AutoARIMA",
    "PolynomialTrendForecaster",
    "SARIMAX",
]
AVAIL_METRICS = [
    "Mean Absolute Error",
    "Root Mean Squared Error",
    "Symmetric Mean Absolute Percentage Error",
    "Symmetric Mean Squared Percentage Error",
    "Median Squared Error",
    "Median Absolute Error",
    "Median Absolute Percentage Error",
    "Median Squared Percentage Error",
]
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

AVAIL_MODELS_GRID = {
    "Prophet": {
        "estimator__forecaster__seasonality_mode": ["additive", "multiplicative"],
    },
    "PolynomialTrendForecaster": {
        "estimator__forecaster__degree": [1, 2, 3],
    },
    "AutoARIMA": {
        "estimator__forecaster__start_p": [1, 2, 3],
        "estimator__forecaster__start_q": [1, 2, 3],
        "estimator__forecaster__sp": [1, 4, 12, 52, 365],
        "estimator__forecaster__seasonal": [True, False],
        "estimator__forecaster__stepwise": [True, False],
        "estimator__forecaster__stationary": [True, False],
        "estimator__forecaster__enforce_stationarity": [True, False],
        "estimator__forecaster__enforce_invertibility": [True, False],
    },
    "SARIMAX": {
        "estimator__forecaster__trend": ["n", "c", "t", "ct"],
        "estimator__forecaster__enforce_stationarity": [True, False],
        "estimator__forecaster__enforce_invertibility": [True, False],
    },
}

AVAIL_TRANSFORMERS_GRID = {
    "Detrender": {
        "estimator__detrender__passthrough": [True, False],
        "estimator__detrender__model": ["multiplicative"],
    },
    "LogTransformer": {
        "estimator__logtransformer__passthrough": [True, False],
        "estimator__logtransformer__offset": [0, 0.1, 1, 10],
        "estimator__logtransformer__scale": [1, 2, 3, 4],
    },
    "ExponentTransformer": {
        "estimator__exponenttransformer__passthrough": [True, False],
    },
    "BoxCoxTransformer": {
        "estimator__boxcoxtransformer__passthrough": [True, False],
        "estimator__boxcoxtransformer__method": ["pearsonr", "mle"],
    },
}
