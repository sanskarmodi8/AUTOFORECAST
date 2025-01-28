"""
AUTOFORECAST - Streamlit Web Application
This module contains the main Streamlit application code for the AUTOFORECAST tool.
It handles data upload, preprocessing, model training, and forecasting visualization.
"""

import csv
import os
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml
import zenml
from sktime.utils.plotting import plot_series

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

# Define base directory
BASE_DIR = Path(__file__).resolve().parent

# Initialize global variables
df = None
target_column = None

# Set environment variables
os.environ["AUTO_OPEN_DASHBOARD"] = "False"


def clear_previous_results():
    """
    Clear all temporary files and results from previous runs.
    This ensures clean state for each new forecast run.
    """
    paths_to_clear = [
        BASE_DIR / DATA_DIR.parent,
        BASE_DIR / Path("temp_image"),
        BASE_DIR / Path(PARAMS_FILE_PATH),
    ]

    for path in paths_to_clear:
        if path.exists():
            try:
                shutil.rmtree(path)
                logger.info(f"Cleared : {path}")
            except Exception as e:
                logger.warning(f"Failed to clear {path}: {e}")


def process_data_upload(data):
    """
    Load and preprocess the uploaded dataset.

    Args:
        data: Uploaded file object from Streamlit

    Returns:
        pandas.DataFrame: Cleaned and processed dataframe
    """
    content = data.getvalue().decode("utf-8")
    delimiter = csv.Sniffer().sniff(content).delimiter
    df = pd.read_csv(data, sep=delimiter)

    # Clean column names
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(" ", "")
    df = df.drop_duplicates()

    return df


def set_datetime_index(df):
    """
    Set the datetime index based on available datetime columns.

    Args:
        df: Input DataFrame

    Returns:
        pandas.DataFrame: DataFrame with datetime index
    """
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns and "time" in df.columns:
        df["time"] = df["time"].str.replace(".", ":")
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="mixed")
        df = df.drop(["date", "time"], axis=1)
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])
        df = df.drop("date", axis=1)
    elif "time" in df.columns:
        df["time"] = df["time"].str.replace(".", ":")
        df["datetime"] = pd.to_datetime(df["time"])
        df = df.drop("time", axis=1)
    else:
        datetime = df.columns[0]
        df["datetime"] = pd.to_datetime(df[datetime])
        if datetime != "datetime":
            df = df.drop(datetime, axis=1)

    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)
    return df


def save_final_dataset(df, target_column):
    """
    Save the processed dataset for forecasting.

    Args:
        df: Processed DataFrame
        target_column: Name of the target variable
    """
    y = df[target_column]
    create_directories([BASE_DIR / DATA_DIR])
    y.to_csv(f"{BASE_DIR / DATA_DIR}/y.csv")


def load_results(_run_id):
    """
    Load and cache results from the forecasting run.

    Args:
        _run_id: Unique identifier for the current run (used for cache busting)

    Returns:
        dict: Dictionary containing all results and paths
    """
    config = ConfigurationManager()
    forecasting_config = config.get_forecasting_config()
    eval_config = config.get_model_evaluation_config()
    train_config = config.get_preprocessing_and_training_config()

    return {
        "best_params": load_json(BASE_DIR / Path(train_config.best_params)),
        "scores": load_json(BASE_DIR / Path(eval_config.scores)),
        "forecast_vs_actual_plot": BASE_DIR / Path(eval_config.forecast_vs_actual_plot),
        "forecasted_values": pd.read_csv(
            BASE_DIR / Path(forecasting_config.forecast_data)
        ),
        "forecast_plot": BASE_DIR / Path(forecasting_config.forecast_plot),
        "model_path": BASE_DIR / Path(train_config.model),
    }


# Initialize session state variables
if "run_id" not in st.session_state:
    st.session_state.run_id = 0
    clear_previous_results()

if "flag" not in st.session_state:
    st.session_state.flag = False

# Streamlit Interface
st.title("AUTOFORECAST")
st.write(" ")

# File Upload Section
data = st.file_uploader("Upload your dataset", type=["csv", "txt"])
if data is not None:
    df = process_data_upload(data)
    df = set_datetime_index(df)

    # Target Column Selection
    target_column = st.selectbox(
        "Select the target column:",
        df.columns.tolist(),
        help="Target column is the column you want to forecast.",
        placeholder="Select Target Column",
    )
    df = df[[target_column]]

    st.markdown("---")

    # Data Preview
    st.write("Data preview:")
    st.dataframe(df.head())

    # Plot Target Column
    create_directories([BASE_DIR / Path("temp_image")])
    plot_series(df[target_column], labels=[target_column])
    plt.savefig(BASE_DIR / Path("temp_image/target_column_plot.png"))
    st.image(BASE_DIR / Path("temp_image/target_column_plot.png"))
    st.write(" ")

    st.markdown("---")

# Forecasting Parameters Section
fh = st.number_input(
    "Select forecast horizon (fh):",
    min_value=1,
    help="Forecast horizon is the number of time steps to forecast into the future.",
    value=7,
)

# Transformer Selection
transformations = st.multiselect(
    "Select Transformer(s)",
    AVAIL_TRANSFORMERS,
    placeholder="Select Transformer(s)",
    help="Transformers are used to preprocess the data before forecasting. You can select none or multiple transformers.",
)
if len(transformations) > 1:
    st.info(
        "⚠️\nThe more transformers you select, the more time it will take to forecast."
    )

# Model Selection
models = st.multiselect(
    "Select Model(s)",
    AVAIL_MODELS,
    placeholder="Select Model(s)",
    help="Models are used to forecast the target variable. Multiple models can be selected.",
)
if len(models) > 1:
    st.info("⚠️\nThe more models you select, the more time it will take to forecast.")

# Metric Selection
metrics = st.multiselect(
    "Select Metric(s)",
    AVAIL_METRICS,
    placeholder="Select Metric(s)",
    help="Metrics are used to evaluate model performance. Multiple metrics can be selected.",
)

st.write(" ")

# Forecast Button and Processing
if st.button("Forecast"):

    # Clear previous results
    clear_previous_results()

    # Increment run ID to force new execution
    st.session_state.run_id += 1

    # Validate user inputs
    if len(models) == 0 or len(metrics) == 0:
        st.error("Please select at least one model and metric.")
        st.stop()

    # Save parameters
    params = {
        "chosen_transformers": transformations,
        "chosen_models": models,
        "chosen_metrics": metrics,
        "fh": fh,
        "run_id": st.session_state.run_id,
    }

    create_directories([BASE_DIR / Path(PARAMS_FILE_PATH).parent])
    save_yaml(params, BASE_DIR / PARAMS_FILE_PATH)

    # Save final dataset
    save_final_dataset(df, target_column)

    # Run forecasting pipeline
    with st.spinner(
        "Hang tight! Forecasting might take a while. Perfect time to grab a ☕"
    ):
        try:
            os.system("zenml up")
            os.system("zenml init")
            run = forecasting_pipeline()
            logger.info("Zenml pipeline run :- \n\n{}".format(run))
            st.success("Forecasting completed!")
            st.session_state.flag = True
        except Exception as e:
            st.error(f"Error during forecasting: {str(e)}")
            logger.error(f"Forecasting pipeline error: {e}")
            clear_previous_results()
            raise e

# Results Display Section
if st.session_state.flag:
    try:
        # Load results with cache busting
        results = load_results(st.session_state.run_id)

        # Display best parameters and model
        st.write("Best params and forecaster chosen:")
        st.json(results["best_params"])

        # Display evaluation results
        st.write("Evaluation results:")
        st.json(results["scores"])
        st.image(
            results["forecast_vs_actual_plot"],
            caption="Forecast vs Actual Plot (for the test data)",
        )

        # Display forecasted values
        st.write("Forecasted values:")
        st.dataframe(results["forecasted_values"])
        st.image(results["forecast_plot"], caption="Forecast Plot (based on given fh)")

        # Model download option
        with open(results["model_path"], "rb") as f:
            st.download_button(
                label="Download the final model",
                data=f,
                file_name="model.joblib",
                mime="application/octet-stream",
            )

    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        logger.error(f"Error loading results: {e}")
        clear_previous_results()
        raise e

# Footer with Issue Reporting Link
st.markdown("---")
st.markdown(
    "Facing an issue or have a feature request? "
    "[Open an issue](https://github.com/sanskarmodi8/AUTOFORECAST/issues) "
    "on our GitHub repository."
)
