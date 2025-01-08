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
    "RandomForestForecaster",
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
        "forecaster__strategy": ["last", "mean", "seasonal"],
    },
    "AutoARIMA": {
        "forecaster__sp": [4, 6, 12],
        "forecaster__stationary": [True, False],
        "forecaster__max_order": [5, 7],
        "forecaster__max_iter": [50, 100],
        "forecaster__start_p": [1, 2],
        "forecaster__start_q": [1, 2],
        "forecaster__d": [0, 1, 2],
    },
    "ExponentialSmoothing": {
        "forecaster__trend": ["add", "mul"],
        "forecaster__seasonal": ["add", "mul"],
        "forecaster__damped_trend": [True, False],
        "forecaster__sp": [4, 6, 12, 30, 48, 365],
    },
    "Prophet": {
        "forecaster__seasonality": ["add", "mul"],
        "forecaster__changepoint_prior_scale": [0.01, 0.1, 1],
        "forecaster__holidays": [None, "US"],
        "forecaster__yearly_seasonality": [True, False],
    },
    "ThetaForecaster": {
        "forecaster__sp": [4, 6, 12],
        "forecaster__theta": [0.1, 0.3, 0.5],
    },
    "SeasonalNaiveForecaster": {
        "forecaster__sp": [4, 6, 12],
    },
    "PolynomialTrendForecaster": {
        "forecaster__degree": [1, 2, 3],
        "forecaster__trend": ["add", "mul"],
    },
    "ARIMA": {
        "forecaster__p": [1, 2],
        "forecaster__d": [1, 2],
        "forecaster__q": [1, 2],
        "forecaster__seasonal_order": [(1, 1, 1, 12)],
        "forecaster__max_iter": [50, 100],
        "forecaster__method": ["mle", "css"],
    },
    "SARIMA": {
        "forecaster__p": [1, 2],
        "forecaster__d": [1],
        "forecaster__q": [1, 2],
        "forecaster__seasonal_order": [
            (P, D, Q, s)
            for P in [0, 1]
            for D in [0, 1]
            for Q in [0, 1]
            for s in [7, 12]
        ],
        "forecaster__trend": ["n", "c", "t", "ct"],
    },
    "SARIMAX": {
        "forecaster__p": [1, 2],
        "forecaster__d": [1],
        "forecaster__q": [1, 2],
        "forecaster__seasonal_order": [(1, 1, 1, 12)],
        "forecaster__exog": [True, False],
        "forecaster__max_iter": [50, 100],
    },
    "RandomForestForecaster": {
        "forecaster__n_estimators": [50, 100, 200],
        "forecaster__max_depth": [10, 20, None],
        "forecaster__min_samples_split": [2, 5],
        "forecaster__min_samples_leaf": [1, 2],
    },
}

AVAIL_TRANSFORMERS_GRID = {
    "Detrender": {
        "estimator__detrender__passthrough": [True, False],
    },
    "LogTransformer": {
        "estimator__logtransformer__passthrough": [True, False],
    },
    "ExponentTransformer": {
        "estimator__exponenttransformer__passthrough": [True, False],
    },
    "Imputer": {
        "estimator__imputer__strategy": ["mean", "median", "most_frequent", "constant"],
        "estimator__imputer__fill_value": [0, 1, -1, "missing"],
    },
    "MinMaxScaler": {
        "estimator__scaler__transformer__with_scaling": [True, False],
        "estimator__scaler__transformer__with_centering": [True, False],
    },
    "Deseasonalizer": {
        "estimator__deseasonalizer__passthrough": [True, False],
        "estimator__deseasonalizer__sp": [4, 6, 12, 30, 48, 365],
    },
    "StandardScaler": {
        "estimator__scaler__transformer__with_scaling": [True, False],
        "estimator__scaler__transformer__with_centering": [True, False],
    },
    "PowerTransformer": {
        "estimator__powertransformer__method": ["yeo-johnson", "box-cox"],
        "estimator__powertransformer__standardize": [True, False],
    },
    "RobustScaler": {
        "estimator__scaler__passthrough": [True, False],
        "estimator__scaler__transformer__transformer__with_scaling": [True, False],
        "estimator__scaler__transformer__transformer__with_centering": [True, False],
    },
}
