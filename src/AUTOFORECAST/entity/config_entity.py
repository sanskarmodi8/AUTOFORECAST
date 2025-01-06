from dataclasses import dataclass
from pathlib import Path

from box import ConfigBox


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    raw_data_dir: Path
    train_data_dir: Path
    val_data_dir: Path
    test_data_dir: Path
    chosen_transformers: list


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_data_dir: Path
    val_data_dir: Path
    model: Path
    chosen_models: list


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model: Path
    test_data_dir: Path
    scores: Path
    forecast_vs_actual_plot: Path
    chosen_metrics: list


@dataclass(frozen=True)
class ForecastingConfig:
    root_dir: Path
    model: Path
    forecast_plot: Path
    fh: int
