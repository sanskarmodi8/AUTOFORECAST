from zenml.steps import step

from AUTOFORECAST import logger
from AUTOFORECAST.components.forecasting import Forecasting
from AUTOFORECAST.config.configuration import ConfigurationManager

STAGE_NAME = "FORECASTING STAGE"


@step(enable_cache=False)
def forecast_step(success: bool) -> None:
    """
    A ZenML step that makes forecasts using the model trained in the previous step.

    This step will load the saved model, use the given fh, and make forecasts using the saved model.
    """

    try:
        if not success:
            logger.error("MODEL EVALUATION STAGE FAILED. SKIPPING FORECASTING STAGE")
        logger.info(f"{STAGE_NAME} STARTED")
        config_manager = ConfigurationManager()
        forecasting = Forecasting(config_manager.get_forecasting_config())
        forecasting.forecast()
        logger.info(f"{STAGE_NAME} COMPLETED")
    except Exception as e:
        logger.error(f"{STAGE_NAME} FAILED: {e}")
        raise e
