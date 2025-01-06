import csv

import pandas as pd
import streamlit as st

X, y = None, None
DATA_DIR = "artifacts/data"

st.title("AUTOFORECAST")

# read dataset
data = st.file_uploader("Upload your dataset", type=["csv"])
if data is not None:
    content = data.getvalue().decode("utf-8")
    delimeter = csv.Sniffer().sniff(content).delimiter
    df = pd.read_csv(data, sep=delimeter)
    df.columns = df.columns.str.lower()

    # set datetime index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    elif 'date' in df.columns and 'time' in df.columns:
        df['time'] = df['time'].str.replace('.', ':')
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='mixed')
        df = df.drop(['date', 'time'], axis=1)
        df.set_index('datetime', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif 'time' in df.columns:
        df['time'] = df['time'].str.replace('.', ':')
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

    # select endogenous variable
    target_column = st.selectbox(
        "Select the target column:",
        df.columns.tolist(),
        help="The target column is the variable you want to forecast.",
        placeholder="Select target column",
    )

    # univariate or multivariate
    isunivariate = st.radio(
        "Are you performing univariate or multivariate analysis?",
        ["Univariate", "Multivariate"],
        help="Univariate analysis involves forecasting a single variable by itself, while multivariate analysis involves forecasting the target variable based on other independent variables in the dataset.",
    )
    if isunivariate == "Multivariate":
        # select exogenous variables
        exogenous_columns = st.multiselect(
            "Select exogenous/independent column(s):",
            df.columns.tolist(),
            help="Exogenous columns are used to forecast the target variable.",
            placeholder="Select exogenous column(s)",
        )
        df = df[[target_column] + exogenous_columns]
        st.write("Dataset preview :")
        st.dataframe(df)
        
        # select fh
        fh = st.number_input(
            "Select the forecast horizon:",
            min_value=1,
            help="The forecast horizon is the number of steps ahead to forecast.",
        )

    y = df[target_column]
    y.to_csv(f"{DATA_DIR}/y.csv")
    if isunivariate == "Multivariate":
        X = df.drop(target_column, axis=1)
        X.to_csv(f"{DATA_DIR}/X.csv")


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
    "Select Transformer(s)", options, placeholder="Select Transformer(s)",
    help="Transformers are used to preprocess the data before forecasting.",
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
models = st.multiselect("Select Model(s)", options, placeholder="Select Model(s)", help="Models are used to forecast the target variable.")

# select metric(s)
options = ["mae", "mse", "rmse", "mape", "smape", "mdape", "r2", "exp_var", "max_error"]
metrics = st.multiselect("Select Metric(s)", options, placeholder="Select Metric(s)", help="Metrics are used to evaluate the performance of the model.")

# forecast
if st.button("Forecast"):
    st.write("Forecasting...")
