from logging import root
from dataclasses import dataclass # No need of constructor for a class
from pathlib import Path

@dataclass( frozen=True) # No changes can be made externally
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataPreparationConfig:
    root_dir: Path
    data_dir: Path
    prepared_dir: Path
    splits: list
    label_prefixes: dict


@dataclass(frozen=True)
class PrepareBaselineModelConfig:
    root_dir: Path
    baseline_model_path: Path
    updated_baseline_model_path: Path
    params_augmentation: bool
    params_image_size: list
    params_classes: int
    params_batch_size: int
    params_include_top: bool
    params_epochs: int
    params_weights: str
    params_learning_rate: float

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    training_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    param_epochs: int
    params_patience: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_learning_rate: float
