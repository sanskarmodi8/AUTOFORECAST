import csv

import pandas as pd
import streamlit as st

from src.AUTOFORECAST.constants import (
    AVAIL_METRICS,
    AVAIL_MODELS,
    AVAIL_TRANSFORMERS,
    DATA_DIR,
)
from src.AUTOFORECAST.pipeline.prediction import Forecast


def process_data_upload(data):
    """Process the uploaded dataset."""
    content = data.getvalue().decode("utf-8")
    delimiter = csv.Sniffer().sniff(content).delimiter
    df = pd.read_csv(data, sep=delimiter)
    df.columns = df.columns.str.lower()
    return df


def set_datetime_index(df):
    """Set the datetime index based on available columns."""
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
    elif "date" in df.columns and "time" in df.columns:
        df["time"] = df["time"].str.replace(".", ":")
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="mixed")
        df = df.drop(["date", "time"], axis=1)
        df.set_index("datetime", inplace=True)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif "time" in df.columns:
        df["time"] = df["time"].str.replace(".", ":")
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    return df


def save_data(df, target_column, is_univariate):
    """Save the target (y) and exogenous (X) variables to CSV."""
    y = df[target_column]
    y.to_csv(f"{DATA_DIR}/y.csv")
    if not is_univariate:
        X = df.drop(target_column, axis=1)
        X.to_csv(f"{DATA_DIR}/X.csv")


# Streamlit Interface
st.title("AUTOFORECAST")

# File Upload
data = st.file_uploader("Upload your dataset", type=["csv"])

if data is not None:
    df = process_data_upload(data)
    df = set_datetime_index(df)

    # Select Target Column
    target_column = st.selectbox("Select the target column:", df.columns.tolist(), help="Target column is the column you want to forecast.", placeholder="Select Target Column")

    # Select Univariate or Multivariate
    is_univariate = st.radio(
        "Univariate or Multivariate?", ["Univariate", "Multivariate"], help="Univariate forecasting involves forecasting the target variable by itself, while multivariate forecasting involves forecasting the target variable using other indipendent/exogenous variables."
    )
    if is_univariate == "Multivariate":
        exogenous_options = [col for col in df.columns if col != target_column]
        exogenous_columns = st.multiselect(
            "Select exogenous columns:", exogenous_options,
            help="Exogenous columns are independent variables used to forecast the target variable.",
            placeholder="Select Exogenous Columns"
        )
        df = df[[target_column] + exogenous_columns]
    else:
        df = df[[target_column]]

    # Display Dataset
    st.write("Dataset preview:")
    st.dataframe(df)

    # Forecast Horizon
    global fh
    fh = st.number_input("Select forecast horizon:", min_value=1, help="Forecast horizon is the number of time steps to forecast into the future.",)

    # Save data to CSV for future processing
    save_data(df, target_column, is_univariate)

# Select Transformer(s)
transformations = st.multiselect(
    "Select Transformer(s)",
    AVAIL_TRANSFORMERS,
    placeholder="Select Transformer(s)",
    help="Transformers are used to preprocess the data before forecasting.",
)

# Select Model(s)
models = st.multiselect(
    "Select Model(s)",
    AVAIL_MODELS,
    placeholder="Select Model(s)",
    help="Models are used to forecast the target variable.",
)

# Select Metric(s)
metrics = st.multiselect(
    "Select Metric(s)",
    AVAIL_METRICS,
    placeholder="Select Metric(s)",
    help="Metrics are used to evaluate the performance of the model.",
)

# Forecast Button
if st.button("Forecast"):
    with st.spinner("Forecasting..."):
        pass
