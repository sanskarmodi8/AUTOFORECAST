import pandas as pd
import streamlit as st

st.title("AUTOFORECAST")

# upload dataset
csv = st.file_uploader("Upload your dataset", type=["csv"])
if csv is not None:
    df = pd.read_csv(csv)
    st.write("Dataset Preview :")
    st.dataframe(df)

# select transformation(s)
options = [
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
transformations = st.multiselect(
    "Select Transformer(s)", options, placeholder="Select Transformer(s)"
)

# select model(s)
options = [
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
models = st.multiselect("Select Model(s)", options, placeholder="Select Model(s)")

# select metric(s)
options = ["mae", "mse", "rmse", "mape", "smape", "mdape", "r2", "exp_var", "max_error"]
metrics = st.multiselect("Select Metric(s)", options, placeholder="Select Metric(s)")

# forecast
if st.button("Forecast"):
    st.write("Forecasting...")
