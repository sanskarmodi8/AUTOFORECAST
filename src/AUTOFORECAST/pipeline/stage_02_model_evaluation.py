from zenml.steps import step

from AUTOFORECAST import logger
from AUTOFORECAST.components.model_evaluation import ModelEvaluation
from AUTOFORECAST.config.configuration import ConfigurationManager

STAGE_NAME = "MODEL EVALUATION STAGE"


@step(enable_cache=False)
def evaluate_step(success: bool) -> bool:
    """
    A ZenML step that evaluates the model performance using the chosen metrics.

    This step will load the saved model, load the test data, and evaluate the model using the chosen metrics.
    """

    try:
        if not success:
            logger.error(
                "PREPROCESSING AND MODEL TRAINING STAGE FAILED. SKIPPING OTHER STAGES"
            )
            return False
        logger.info(f"{STAGE_NAME} STARTED")
        config = ConfigurationManager().get_model_evaluation_config()
        ModelEvaluation(config).evaluate()
        logger.info(f"{STAGE_NAME} COMPLETED")
        return True
    except Exception as e:
        logger.error(f"{STAGE_NAME} FAILED: {e}")
        raise e
        return False
