from pathlib import Path

DATA_DIR = Path("artifacts/data")
AVAIL_TRANSFORMERS = [
    "Detrender",
    "LogTransformer",
    "ExponentTransformer",
    "Imputer",
    "Deseasonalizer",
    "PowerTransformer",
    "RobustScaler",
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
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "mdape",
    "r2",
    "exp_var",
    "max_error",
]
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

AVAIL_MODELS_GRID = {
    "NaiveForecaster": {
        "estimator__forecaster__strategy": ["last", "mean", "drift"],
        "estimator__forecaster__sp": [1, 4, 12, 365],
    },
    "AutoARIMA": {
        "estimator__forecaster__sp": [12],
        "estimator__forecaster__stationary": [True, False],
    },
    "ExponentialSmoothing": {
        "estimator__forecaster__trend": ["add", "mul"],
        "estimator__forecaster__seasonal": ["add", "mul"],
        "estimator__forecaster__damped_trend": [True, False],
        "estimator__forecaster__sp": [1, 4, 12, 365],
        "estimator__forecaster__use_boxcox": [True, False, "log", float],
        "estimator__forecaster__remove_bias": [True, False],
    },
    "Prophet": {
        "estimator__forecaster__seasonality_mode": ["additive", "multiplicative"],
    },
    "ThetaForecaster": {
        "estimator__forecaster__sp": [1, 4, 12, 365],
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
    "Imputer": {
        "estimator__imputer__method": [
            "drift",
            "median",
            "nearest",
            "mean",
        ],
        "estimator__imputer__passthrough": [True, False],
    },
    "Deseasonalizer": {
        "estimator__deseasonalizer__passthrough": [True, False],
        "estimator__deseasonalizer__sp": [1, 4, 12, 365],
    },
    "PowerTransformer": {
        "estimator__powertransformer__method": ["yeo-johnson", "box-cox"],
        "estimator__powertransformer__standardize": [True, False],
        "estimator__powertransformer__passthrough": [True, False],
    },
    "RobustScaler": {
        "estimator__scaler__passthrough": [True, False],
        "estimator__scaler__with_scaling": [True, False],
        "estimator__scaler__with_centering": [True, False],
    },
}
