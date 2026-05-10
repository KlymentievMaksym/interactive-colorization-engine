import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    PathType = str | list[str]
else:
    PathType = Any

@dataclass
class ModelNode:
    model_name: str = "mamba"
    weights: str | None = None
    model_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LossNode:
    loss_name: str = "colorization"
    loss_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 4
    timeout: float=  60

    min_hint_size: int = 2
    max_hint_size: int = 16
    num_hints_val: int = 3
    patch_size_val: int = 15

    resume: str | None = None
    do_save: bool = True
    amount_show: int = 4
    val_check_interval: int | float = 0.5

@dataclass
class DataConfig:
    train: list[PathType] = field(default_factory=list)
    val: list[PathType] | None = None
    test: list[PathType] | None = None

@dataclass
class MainConfig:
    image_size: int = 256
    device: str | None = None
    model: ModelNode = field(default_factory=ModelNode)

@dataclass
class TrainConfig(MainConfig):
    loss: LossNode = field(default_factory=LossNode)
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