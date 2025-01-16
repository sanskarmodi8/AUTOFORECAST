from zenml.pipelines import pipeline
from AUTOFORECAST.pipeline.stage_01_preprocessing_and_training import (
    preprocess_and_train_step,
)
from AUTOFORECAST.pipeline.stage_02_model_evaluation import evaluate_step
from AUTOFORECAST.pipeline.stage_03_forecasting import forecast_step


@pipeline
def forecasting_pipeline():
    """
    A ZenML pipeline for forecasting that executes three main steps:
    1. Preprocessing and Training: Prepares the data and trains the model.
    2. Evaluation: Evaluates the model's performance using specified metrics.
    3. Forecasting: Generates forecasts based on the trained model.

    Each step is represented by a function passed as an argument to the pipeline.
    """

    success = preprocess_and_train_step()
    success2 = evaluate_step(success)
    forecast_step(success2)
