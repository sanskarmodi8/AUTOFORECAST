from zenml.pipelines import pipeline

from AUTOFORECAST.pipeline.stage_01_data_analysis import data_analysis_step
from AUTOFORECAST.pipeline.stage_02_preprocessing_and_training import (
    preprocess_and_train_step,
)
from AUTOFORECAST.pipeline.stage_03_model_evaluation import evaluate_step
from AUTOFORECAST.pipeline.stage_04_forecasting import forecast_step


@pipeline
def forecasting_pipeline():
    """
    A ZenML pipeline that orchestrates the forecasting process.

    This pipeline consists of four steps:
    1. Data Analysis: Analyzes the input data and ensures it is ready for preprocessing.
    2. Preprocessing and Training: Preprocesses the data and trains the model.
    3. Model Evaluation: Evaluates the trained model's performance.
    4. Forecasting: Uses the trained model to make future forecasts.

    Each step depends on the success of the previous step, ensuring a sequential workflow.
    """

    success = data_analysis_step()
    success2 = preprocess_and_train_step(success)
    success3 = evaluate_step(success2)
    forecast_step(success3)
