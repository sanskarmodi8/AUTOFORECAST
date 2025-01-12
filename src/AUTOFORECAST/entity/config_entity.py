from dataclasses import dataclass
from pathlib import Path

from box import ConfigBox

# entity classes for final configuration of each component of the pipeline


@dataclass(frozen=True)
class PreprocessingAndTrainingConfig:
    root_dir: Path
    model: Path
    test_data_dir: Path
    chosen_transformers: list
    chosen_models: list
    best_params: Path


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model: Path
    test_data_dir: Path
    scores: Path
    forecast_vs_actual: Path
    chosen_metrics: list


@dataclass(frozen=True)
class ForecastingConfig:
    root_dir: Path
    model: Path
    forecast_plot: Path
    fh: int
