from zenml.steps import step

from AUTOFORECAST import logger
from AUTOFORECAST.components.data_analysis import DataAnalysis
from AUTOFORECAST.config.configuration import ConfigurationManager

STAGE_NAME = "DATA ANALYSIS STAGE"


@step(enable_cache=False)
def data_analysis_step() -> bool:
    """
    A ZenML step that runs data analysis.

    This step will load the configuration for data analysis, run the data analysis
    step, and return a boolean indicating whether the step was successful or not.

    Returns:
        bool: True if the step was successful, False otherwise.
    """
    try:
        logger.info(f"{STAGE_NAME} STARTED")
        config_manager = ConfigurationManager()
        config = config_manager.get_data_analysis_config()
        data_analysis = DataAnalysis(config)
        data_analysis.analyze()
        logger.info(f"{STAGE_NAME} COMPLETED")
        return True
    except Exception as e:
        logger.error(f"{STAGE_NAME} FAILED: {e}")
        raise e
        return False
