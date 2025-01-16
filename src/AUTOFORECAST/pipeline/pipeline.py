from zenml.pipelines import pipeline


@pipeline
def forecasting_pipeline(preprocess_and_train, evaluate, forecast):
    """
    A ZenML pipeline for forecasting that executes three main steps:
    1. Preprocessing and Training: Prepares the data and trains the model.
    2. Evaluation: Evaluates the model's performance using specified metrics.
    3. Forecasting: Generates forecasts based on the trained model.

    Each step is represented by a function passed as an argument to the pipeline.
    """

    preprocess_and_train()
    evaluate()
    forecast()
