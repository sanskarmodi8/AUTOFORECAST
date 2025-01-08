from zenml.pipelines import pipeline


@pipeline
def forecasting_pipeline(analyze, preprocess_and_train, evaluate, forecast):
    success = analyze()
    success2 = preprocess_and_train(success)
    success3 = evaluate(success2)
    success4 = forecast(success3)
    print("Success of the pipeline:", success4)
