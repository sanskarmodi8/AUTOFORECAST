from zenml.pipelines import pipeline


@pipeline
def forecasting_pipeline(preprocess_and_train):
    preprocess_and_train()
    # evaluate()
