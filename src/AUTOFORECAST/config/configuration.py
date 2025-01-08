from pathlib import Path

from AUTOFORECAST.constants import CONFIG_FILE_PATH, DATA_DIR, PARAMS_FILE_PATH
from AUTOFORECAST.entity.config_entity import (
    DataAnalysisConfig,
    ForecastingConfig,
    ModelEvaluationConfig,
    PreprocessingAndTrainingConfig,
)
from AUTOFORECAST.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(self):
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.config = read_yaml(CONFIG_FILE_PATH)

    def get_data_analysis_config(self) -> DataAnalysisConfig:
        config = self.config.data_analysis
        create_directories([Path(config.root_dir)])
        return DataAnalysisConfig(
            root_dir=config.root_dir, data_summary=config.data_summary
        )

    def get_preprocessing_and_training_config(self) -> PreprocessingAndTrainingConfig:
        config = self.config.preprocessing_and_training
        create_directories([Path(config.root_dir), Path(config.test_data_dir)])
        return PreprocessingAndTrainingConfig(
            root_dir=config.root_dir,
            model=config.model,
            test_data_dir=config.test_data_dir,
            chosen_transformers=self.params.chosen_transformers,
            chosen_models=self.params.chosen_models,
            data_summary=config.data_summary,
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
