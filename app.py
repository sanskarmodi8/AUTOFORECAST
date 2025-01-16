import csv
import os

import pandas as pd
import streamlit as st
import yaml
import zenml

from src.AUTOFORECAST import logger
from src.AUTOFORECAST.constants import (
    AVAIL_METRICS,
    AVAIL_MODELS,
    AVAIL_TRANSFORMERS,
    DATA_DIR,
    PARAMS_FILE_PATH,
)
from src.AUTOFORECAST.pipeline.pipeline import forecasting_pipeline
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
        df.sort_values("datetime", inplace=True)
        df.set_index("datetime", inplace=True)
    return df


def save_final_dataset(df, target_column, with_or_without):
    """Save the CSV data in appropiate manner."""
    y = df[target_column]
    create_directories([DATA_DIR])
    y.to_csv(f"{DATA_DIR}/y.csv")
    if with_or_without == "With Exogenous Data":
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

    # Forecast Horizon
    fh = st.number_input(
        "Select forecast horizon (fh):",
        min_value=1,
        help="Forecast horizon is the number of time steps to forecast into the future.",
        value=7,
    )
    st.write("")

    if len(df.columns)>1:
        # choice for exog data
        with_or_without = st.radio(
            "You want to do forecasting with or without exogenous variables?",
            ["Without Exogenous Data", "With Exogenous Data"],
            help="Exogenous variables are independent variables used to forecast the target variable.",
        )

        if with_or_without == "With Exogenous Data":
            st.info(
                "Please make sure that you have at least 'fh' number of future values of the exogenous variables. This is required for forecasting if the model is trained using Exogenous Data. Otherwise, you should select 'Without Exogenous Data'."
            )
            exogenous_options = [col for col in df.columns if col != target_column]
            exogenous_columns = st.multiselect(
                "Select exogenous columns:",
                exogenous_options,
                placeholder="Select Exogenous Columns",
            )
            df = df[[target_column] + exogenous_columns]
        else:
            df = df[[target_column]]
    else:
        with_or_without = "Without Exogenous Data"

    # Display Dataset
    st.write("")
    st.write("Dataset preview:")
    st.dataframe(df)

    # Save data to CSV for future processing
    save_final_dataset(df, target_column, with_or_without)

# Select Transformer(s)
transformations = st.multiselect(
    "Select Transformer(s)",
    AVAIL_TRANSFORMERS,
    placeholder="Select Transformer(s)",
    help="Transformers are used to preprocess the data before forecasting. You can select none of the transformer or multiple transformers. We will take care of the best possible ordering, if you selected more than one.",
)
if len(transformations) > 1:
    st.info(
        "Just remember, the more transformers you select, the more time it will take to forecast, or the app may even crash if the memory exceeds."
    )

# Select Model(s)
models = st.multiselect(
    "Select Model(s)",
    AVAIL_MODELS,
    placeholder="Select Model(s)",
    help="Models are used to forecast the target variable. You can select multiple models, we will choose the best performing one.",
)
if len(models) > 1:
    st.info(
        "Just remember, the more models you select, the more time it will take to forecast, or the app may even crash if the memory exceeds."
    )

# Select Metric(s)
metrics = st.multiselect(
    "Select Metric(s)",
    AVAIL_METRICS,
    placeholder="Select Metric(s)",
    help="Metrics are used to evaluate the performance of the model. You can select multiple metrics, we will evaluate on each one of what you selected.",
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
        run = forecasting_pipeline()
        logger.info(f"Zenml Pipeline run: {run}")
        # TODO: Display the results
        st.success("Forecasting completed!")
