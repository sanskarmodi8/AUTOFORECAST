from pathlib import Path

DATA_DIR = Path("artifacts/data")
AVAIL_TRANSFORMERS = [
    "LogTransformer",
    "ExponentTransformer",
]
AVAIL_MODELS = [
    "Prophet",
    "PolynomialTrendForecaster",
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
}

AVAIL_TRANSFORMERS_GRID = {
    "LogTransformer": {
        "estimator__logtransformer__passthrough": [True, False],
        "estimator__logtransformer__offset": [0, 0.1],
        "estimator__logtransformer__scale": [1, 2],
    },
    "ExponentTransformer": {
        "estimator__exponenttransformer__passthrough": [True, False],
    },
}
