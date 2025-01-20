import csv
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import yaml
import zenml

from src.AUTOFORECAST import logger
from src.AUTOFORECAST.config.configuration import ConfigurationManager
from src.AUTOFORECAST.constants import (
    AVAIL_METRICS,
    AVAIL_MODELS,
    AVAIL_TRANSFORMERS,
    DATA_DIR,
    PARAMS_FILE_PATH,
)
from src.AUTOFORECAST.pipeline.pipeline import forecasting_pipeline
from src.AUTOFORECAST.utils.common import create_directories, load_json, save_yaml

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


def save_final_dataset(df, target_column):
    """Save the CSV data in appropiate manner."""
    y = df[target_column]
    create_directories([DATA_DIR])
    y.to_csv(f"{DATA_DIR}/y.csv")


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
    df = df[[target_column]]

    # Forecast Horizon
    fh = st.number_input(
        "Select forecast horizon (fh):",
        min_value=1,
        help="Forecast horizon is the number of time steps to forecast into the future.",
        value=7,
    )
    st.write("")

    # Display Dataset
    st.write("Dataset preview:")
    st.dataframe(df)

    # Save data to CSV for future processing
    save_final_dataset(df, target_column)

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

# Initialize session state for the flag
if "flag" not in st.session_state:
    st.session_state.flag = False

# Forecast Button
if st.button("Forecast"):
    # check if the user provided data is valid
    if len(models) == 0 or len(metrics) == 0:
        st.error("Please select at least one model, and metric.")
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
        logger.info("Zenml pipeline run :- \n\n{}".format(run))
        st.success("Forecasting completed!")
        st.session_state.flag = True

if st.session_state.flag:
    # show the results
    # load the configurations to get the output paths
    config = ConfigurationManager()
    forecasting_config = config.get_forecasting_config()
    eval_config = config.get_model_evaluation_config()
    train_config = config.get_preprocessing_and_training_config()

    # best params and forecaster chosen
    st.write("Best params and forecaster chosen:")
    st.json(load_json(Path(train_config.best_params)))

    # evaluation results
    st.write("Evaluation results:")
    st.json(load_json(Path(eval_config.scores)))
    st.image(
        eval_config.forecast_vs_actual_plot,
        caption="Forecast vs Actual Plot (for the test data)",
    )

    # forecasted values
    st.write("Forecasted values:")
    forecasted_values = pd.read_csv(forecasting_config.forecast_data)
    st.dataframe(forecasted_values)
    st.image(
        forecasting_config.forecast_plot,
        caption="Forecast Plot (based on given fh)",
    )

    # final trained model for download
    with open(train_config.model, "rb") as f:
        st.download_button(
            label="Download the final model",
            data=f,
            file_name="model.joblib",
            mime="application/octet-stream",
        )