from zenml.steps import step

from AUTOFORECAST import logger
from AUTOFORECAST.components.preprocessing_and_training import PreprocessingAndTraining
from AUTOFORECAST.config.configuration import ConfigurationManager

STAGE_NAME = "PREPROCESSING AND MODEL TRAINING STAGE"


@step
def preprocess_and_train_step() -> bool:
    try:
        logger.info(f"{STAGE_NAME} STARTED")
        config_manager = ConfigurationManager()
        config = config_manager.get_preprocessing_and_training_config()
        preprocessing_and_training = PreprocessingAndTraining(config)
        preprocessing_and_training.run()
        logger.info(f"{STAGE_NAME} COMPLETED")
        return True
    except Exception as e:
        logger.error(f"{STAGE_NAME} FAILED: {e}")
        raise e
