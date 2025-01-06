from zenml.pipelines import pipeline


@pipeline
def forecasting_pipeline(transform, train, evaluate, forecast):
    success = transform()
    success2 = train(success)
    success3 = evaluate(success2)
    success4 = forecast(success3)
