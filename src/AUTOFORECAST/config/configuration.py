from pathlib import Path

from AUTOFORECAST.constants import CONFIG_FILE_PATH, DATA_DIR, PARAMS_FILE_PATH
from AUTOFORECAST.entity.config_entity import (
    DataTransformationConfig,
    ForecastingConfig,
    ModelEvaluationConfig,
    ModelTrainingConfig,
)
from AUTOFORECAST.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(self):
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.config = read_yaml(CONFIG_FILE_PATH)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories(
            [
                Path(config.root_dir),
                Path(config.train_data_dir),
                Path(config.val_data_dir),
                Path(config.test_data_dir),
            ]
        )
        return DataTransformationConfig(
            root_dir=config.root_dir,
            raw_data_dir=DATA_DIR,
            train_data_dir=config.train_data_dir,
            val_data_dir=config.val_data_dir,
            test_data_dir=config.test_data_dir,
            chosen_transformers=self.params.chosen_transformers,
        )

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        create_directories([Path(config.root_dir)])
        return ModelTrainingConfig(
            root_dir=config.root_dir,
            model=config.model,
            train_data_dir=config.train_data_dir,
            val_data_dir=config.val_data_dir,
            chosen_models=self.params.chosen_models,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([Path(config.root_dir)])
        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            model=config.model,
            test_data_dir=config.test_data_dir,
            scores=config.scores,
            forecast_vs_actual_plot=config.forecast_vs_actual_plot,
            chosen_metrics=self.params.chosen_metrics,
        )

    def forecasting_config(self) -> ForecastingConfig:
        config = self.config.forecasting
        create_directories([Path(config.root_dir)])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            model=config.model,
            forecast_plot=config.forecast_plot,
            fh=config.fh,
        )
