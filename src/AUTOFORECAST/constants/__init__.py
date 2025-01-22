from pathlib import Path

DATA_DIR = Path("artifacts/data")
AVAIL_TRANSFORMERS = ["LogTransformer", "ExponentTransformer", "Deseasonalizer"]
AVAIL_MODELS = ["Prophet", "PolynomialTrendForecaster", "AutoARIMA", "NaiveForecaster", "ExponentialSmoothing"]
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
    "NaiveForecaster": {
        "estimator__forecaster__strategy": ["mean", "last", "drift"],
    },
    "Prophet": {
        "estimator__forecaster__seasonality_mode": ["additive", "multiplicative"],
    },
    "PolynomialTrendForecaster": {
        "estimator__forecaster__degree": [1, 2, 3],
    },
    "AutoARIMA": {
        "start_p": [3, 5],
        "start_q": [3, 5],
        "start_P": [1, 2],
        "start_Q": [1, 2],
    },
    "ExponentialSmoothing": {
        "estimator__forecaster__trend": ["add", "mul"],
        "estimator__forecaster__seasonal": ["add", "mul"],
        "estimator__forecaster__use_boxcox": [True, False],
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
    "Deseasonalizer": {
        "estimator__deseasonalizer__passthrough": [True, False],
        "estimator__deseasonalizer__model": ["additive", "multiplicative"],
    },
}
