import torch
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ModelNode:
    model_name: str = "mamba"
    weights: str | None = None
    model_params: dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 4
    resume: str | None = None
    do_save: bool = True

    loss_lambda_smooth: float = 0.5
    loss_lambda_cosine: float = 1.0

@dataclass
class DataConfig:
    train: list[str] = field(default_factory=list)
    val: list[str] | None = None

@dataclass
class MainConfig:
    image_size: int = 256
    device: str | None = None
    model: ModelNode = field(default_factory=ModelNode)

@dataclass
class TrainConfig(MainConfig):
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

@dataclass
class InferenceConfig(MainConfig):
    image: str | None = None
    input_dir: str | None = None
    result_dir: str = "results/inference"

@dataclass
class EvaluateConfig(MainConfig):
    data: DataConfig = field(default_factory=DataConfig)
    batch_size: int = 4
    output_dir: str = "results/eval"
    save_number: int = 50