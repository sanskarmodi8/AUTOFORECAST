from zenml.steps import step

STAGE_NAME = "FORECASTING STAGE"


@step
def forecast_step(success: bool) -> bool:
    return True
