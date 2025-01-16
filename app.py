import csv
import os

import pandas as pd
import streamlit as st
import yaml
import zenml

from src.AUTOFORECAST.constants import (
    AVAIL_METRICS,
    AVAIL_MODELS,
    AVAIL_TRANSFORMERS,
    DATA_DIR,
    PARAMS_FILE_PATH,
)
from src.AUTOFORECAST.pipeline.pipeline import forecasting_pipeline
from src.AUTOFORECAST.pipeline.stage_01_preprocessing_and_training import (
    preprocess_and_train_step,
)
from src.AUTOFORECAST.pipeline.stage_02_model_evaluation import evaluate_step
from src.AUTOFORECAST.pipeline.stage_03_forecasting import forecast_step
from src.AUTOFORECAST.utils.common import create_directories, save_yaml

# set environment variables
os.environ["AUTO_OPEN_DASHBOARD"] = "False"


def process_data_upload(data):
    """Load the uploaded dataset correctly by determining the delimiter."""
    content = data.getvalue().decode("utf-8")
    delimiter = csv.Sniffer().sniff(content).delimiter
    df = pd.read_csv(data, sep=delimiter)
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(" ", "")
    return df


def set_datetime_index(df):
    """Set the datetime index based on available columns."""
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.sort_values("datetime", inplace=True)
        df.set_index("datetime", inplace=True)
    elif "date" in df.columns and "time" in df.columns:
        df["time"] = df["time"].str.replace(".", ":")
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="mixed")
        df = df.drop(["date", "time"], axis=1)
        df.sort_values("datetime", inplace=True)
        df.set_index("datetime", inplace=True)
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])
        df = df.drop("date", axis=1)
        df.sort_values("datetime", inplace=True)
        df.set_index("datetime", inplace=True)
    elif "time" in df.columns:
        df["time"] = df["time"].str.replace(".", ":")
        df["datetime"] = pd.to_datetime(df["time"])
        df = df.drop("time", axis=1)
        df.sort_values("datetime", inplace=True)
        df.set_index("datetime", inplace=True)
    else:
        datetime = df.columns[0]
        df["datetime"] = pd.to_datetime(df[datetime])
        if datetime != "datetime":
            df = df.drop(datetime, axis=1)
        df.sort_values(datetime, inplace=True)
        df.set_index(datetime, inplace=True)
    return df


def save_final_dataset(df, target_column, uni_or_mul):
    """Save the CSV data in appropiate manner."""
    y = df[target_column]
    create_directories([DATA_DIR])
    y.to_csv(f"{DATA_DIR}/y.csv")
    if uni_or_mul == "Multivariate":
        X = df.drop(target_column, axis=1)
        X.to_csv(f"{DATA_DIR}/X.csv")


# Streamlit Interface
st.title("AUTOFORECAST")

# File Upload
data = st.file_uploader("Upload your dataset", type=["csv", "txt"])
if data is not None:
    df = process_data_upload(data)
    # set the datetime index
    df = set_datetime_index(df)

    # Select Target Column
    target_column = st.selectbox(
        "Select the target column:",
        df.columns.tolist(),
        help="Target column is the column you want to forecast.",
        placeholder="Select Target Column",
    )

    # Select Univariate or Multivariate
    uni_or_mul = st.radio(
        "Univariate or Multivariate?",
        ["Univariate", "Multivariate"],
        help="Univariate forecasting involves forecasting the target variable by itself, while multivariate forecasting involves forecasting the target variable using other indipendent/exogenous variables.",
    )
    if uni_or_mul == "Multivariate":
        exogenous_options = [col for col in df.columns if col != target_column]
        exogenous_columns = st.multiselect(
            "Select exogenous columns:",
            exogenous_options,
            help="Exogenous columns are independent variables used to forecast the target variable.",
            placeholder="Select Exogenous Columns",
        )
        df = df[[target_column] + exogenous_columns]
    else:
        df = df[[target_column]]

    # Display Dataset
    st.write("Dataset preview:")
    st.dataframe(df)

    # Save data to CSV for future processing
    save_final_dataset(df, target_column, uni_or_mul)

# Select Transformer(s)
transformations = st.multiselect(
    "Select Transformer(s)",
    AVAIL_TRANSFORMERS,
    placeholder="Select Transformer(s)",
    help="Transformers are used to preprocess the data before forecasting. You can select multiple transformers, we will take care of the best possible ordering. Just remember, the more transformers you select, the more time it will take to forecast, or the app may crash if the memory exceeds.",
)

# Select Model(s)
models = st.multiselect(
    "Select Model(s)",
    AVAIL_MODELS,
    placeholder="Select Model(s)",
    help="Models are used to forecast the target variable. You can select multiple models, we will choose the best performing one. Just remember, the more models you select, the more time it will take to forecast, or the app may crash if the memory exceeds.",
)

# Select Metric(s)
metrics = st.multiselect(
    "Select Metric(s)",
    AVAIL_METRICS,
    placeholder="Select Metric(s)",
    help="Metrics are used to evaluate the performance of the model. You can select multiple metrics, we will evaluate on each one of what you selected.",
)

# Forecast Horizon
fh = st.number_input(
    "Select forecast horizon:",
    min_value=1,
    help="Forecast horizon is the number of time steps to forecast into the future.",
    value=7,
)

# Forecast Button
if st.button("Forecast"):
    # check if the user provided data is valid
    if len(models) == 0 or len(metrics) == 0:
        st.error("Please select at least one transformer, model, and metric.")
        st.stop()
    # save the user provided data in yaml file
    params = {
        "chosen_transformers": transformations,
        "chosen_models": models,
        "chosen_metrics": metrics,
        "fh": fh,
    }
    save_yaml(params, PARAMS_FILE_PATH)
    with st.spinner("Forecasting..."):
        # Run the forecasting pipeline
        os.system("zenml up")
        os.system("zenml init")
        forecasting_pipeline(
            # preprocess_and_train_step(),
            evaluate_step(),
            forecast_step(),
        ).run()
        # TODO: Display the results
        st.success("Forecasting completed!")
