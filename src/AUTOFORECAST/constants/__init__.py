DATA_DIR = "artifacts/data"
AVAIL_TRANSFORMERS = [
    "Detrender",
    "BoxCoxTransformer",
    "ExponentTransformer",
    "Imputer",
    "MinMaxScaler",
    "Deseasonalizer",
    "PowerTransformer",
    "StandardScaler",
    "Differencer",
    "Lagger",
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
    "BoxCoxBiasAdjustedForecaster",
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
