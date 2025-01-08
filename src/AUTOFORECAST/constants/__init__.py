from pathlib import Path

DATA_DIR = Path("artifacts/data")
AVAIL_TRANSFORMERS = [
    "Detrender",
    "LogTransformer",
    "ExponentTransformer",
    "Imputer",
    "MinMaxScaler",
    "Deseasonalizer",
    "StandardScaler",
    "PowerTransformer",
    "RobustScaler",
]
AVAIL_MODELS = [
    "NaiveForecaster",
    "AutoARIMA",
    "ExponentialSmoothing",
    "Prophet",
    "ThetaForecaster",
    "SeasonalNaiveForecaster",
    "PolynomialTrendForecaster",
    "ARIMA",
    "SARIMA",
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
        "forecaster__strategy": ["last", "mean", "drift"],
        "forecaster__sp": [1, 4, 12, 365],
        "forecaster_window_length": [10, 50, 100, None],
    },
    "AutoARIMA": {
        "forecaster__sp": [1, 4, 12, 365],
        "forecaster__stationary": [True, False],
        "forecaster__start_p": [1, 2],
        "forecaster__start_q": [1, 2],
    },
    "ExponentialSmoothing": {
        "forecaster__trend": ["add", "mul"],
        "forecaster__seasonal": ["add", "mul"],
        "forecaster__damped_trend": [True, False],
        "forecaster__sp": [1, 4, 12, 365],
        "forecaster__use_boxcox": [True, False, "log", float],
        "forecaster__remove_bias": [True, False],
    },
    "Prophet": {
        "forecaster__seasonality_mode": ["additive", "multiplicative"],
    },
    "ThetaForecaster": {
        "forecaster__sp": [1, 4, 12, 365],
        "forecaster__deseasonalize": [True, False],
    },
    "PolynomialTrendForecaster": {
        "forecaster__degree": [1, 2, 3],
    },
    "ARIMA": {
        "forecaster__simple_differencing": [True, False],
        "forecaster__enforce_stationarity": [True, False],
        "forecaster__max_iter": [50, 100],
    },
    "SARIMAX": {
        "forecaster__trend": ["n", "c", "t", "ct"],
        "forecaster__simple_differencing": [True, False],
        "forecaster__enforce_stationarity": [True, False],
    },
}

MODELS_WITH_SP_PARAM = [
    "NaiveForecaster",
    "AutoARIMA",
    "ExponentialSmoothing",
    "ThetaForecaster",
]

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
            "mode",
            "nearest",
            "constant",
            "mean",
            "median",
        ],
        "estimator__imputer__passthrough": [True, False],
    },
    "MinMaxScaler": {
        "estimator__scaler__passthrough": [True, False],
    },
    "Deseasonalizer": {
        "estimator__deseasonalizer__passthrough": [True, False],
        "estimator__deseasonalizer__sp": [1, 4, 12, 365],
    },
    "StandardScaler": {
        "estimator__scaler__pass_through": [True, False],
    },
    "PowerTransformer": {
        "estimator__powertransformer__method": ["yeo-johnson", "box-cox"],
        "estimator__powertransformer__standardize": [True, False],
        "estimator__powertransformer__passthrough": [True, False],
    },
    "RobustScaler": {
        "estimator__scaler__passthrough": [True, False],
        "estimator__scaler__transformer__transformer__with_scaling": [True, False],
        "estimator__scaler__transformer__transformer__with_centering": [True, False],
    },
}

TRANSFORMERS_WITH_SP_PARAM = ["Deseasonalizer"]
