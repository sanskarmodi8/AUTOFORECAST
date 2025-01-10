from zenml.steps import step

from AUTOFORECAST import logger
from AUTOFORECAST.components.model_evaluation import ModelEvaluation
from AUTOFORECAST.config.configuration import ConfigurationManager

STAGE_NAME = "MODEL EVALUATION STAGE"


@step
def evaluate_step():
    try:
        logger.info(f"Starting {STAGE_NAME}")
        config = ConfigurationManager().get_model_evaluation_config()
        ModelEvaluation(config).evaluate()
        logger.info(f"Finished {STAGE_NAME}")
    except Exception as e:
        logger.error(f"Failed {STAGE_NAME} : {str(e)}")
        raise e
