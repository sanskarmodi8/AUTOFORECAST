from zenml.steps import step

from AUTOFORECAST.components.data_analysis import DataAnalysis
from AUTOFORECAST.config.configuration import ConfigurationManager

STAGE_NAME = "DATA ANALYSIS STAGE"


@step
def analyze_step() -> bool:
    """
    Step to perform data analysis.

    This step reads the configuration for data analysis, creates an instance of
    DataAnalysis, and calls its execute_analysis method to perform the actual
    analysis.

    Returns:
        bool: Indicates success or failure of the step.
    """
    try:
        logger.info(f"{STAGE_NAME} STARTED")
        config_manager = ConfigurationManager()
        data_analysis_config = config_manager.get_data_analysis_config()
        data_analysis = DataAnalysis(data_analysis_config)
        data_analysis.execute_analysis()
        logger.info(f"{STAGE_NAME} COMPLETED")
        return True
    except Exception as e:
        logger.error(f"{STAGE_NAME} FAILED: {str(e)}")
        raise e
