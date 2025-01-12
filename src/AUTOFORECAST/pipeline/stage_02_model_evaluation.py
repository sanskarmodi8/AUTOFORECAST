from zenml.steps import step

from AUTOFORECAST import logger
from AUTOFORECAST.components.model_evaluation import ModelEvaluation
from AUTOFORECAST.config.configuration import ConfigurationManager

STAGE_NAME = "MODEL EVALUATION STAGE"


@step
def evaluate_step():
    """
    A ZenML step that evaluates the model performance using the chosen metrics.

    This step will load the saved model, load the test data, and evaluate the model using the chosen metrics.
    """
    try:
        logger.info(f"Starting {STAGE_NAME}")
        config = ConfigurationManager().get_model_evaluation_config()
        ModelEvaluation(config).evaluate()
        logger.info(f"Finished {STAGE_NAME}")
    except Exception as e:
        logger.error(f"Failed {STAGE_NAME} : {str(e)}")
        raise e
