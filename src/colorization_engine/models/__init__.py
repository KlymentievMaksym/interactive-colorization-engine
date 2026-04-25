import os
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from colorization_engine.models.util_models import BaseColorizer
from colorization_engine.models.mamba import MambaWrapper
from colorization_engine.models.zhang import ZhangWrapper
# from colorization_engine.models.deoldify import DeOldifyWrapper


@dataclass
class Model:
    model_class: type[BaseColorizer]
    model_config: Path | str | None
    # model_weights: Path | str | None

@dataclass
class TrainingConfig:
    lr: float
    batch_size: int
    epochs: int

    def __post_init__(self):
        if isinstance(self.lr, str):
            self.lr = float(self.lr)

@dataclass
class Config:
    weights: Path | str | None
    image_size: int | None
    training: TrainingConfig | None
    model_params: dict[str, Any]

    def __post_init__(self):
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)

MODEL_REGISTRY = {
    "mamba": Model(model_class=MambaWrapper, model_config="configs/mamba.yaml"),  #model_weights="checkpoints/mamba.pth", 
    # "zhang_siggraph17": Model(model_class=ZhangWrapper, model_weights=None, model_config=None),
    # "deoldify_artistic": Model(model_class=DeOldifyWrapper, model_weights=None, model_config=None),
}

def load_colorization_model(model_name: str, device: torch.device | str, weights_path: Path | str | None = None, config_path: Path | str | None = None, **override_model_params) -> tuple[torch.nn.Module, Config]:
    """Loads models by its name from model registry"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"[ERROR] Model {model_name} not found in model registry. Available: {list(MODEL_REGISTRY.keys())}")

    info = MODEL_REGISTRY[model_name]

    standard_config = None
    config_path = config_path if config_path is not None else info.model_config
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                yaml_params = yaml.safe_load(f) or {}
            yaml_params["model_params"].update(override_model_params)
            standard_config = Config(**yaml_params)
        else:
            print(f"[WARNING] Config file not found: {config_path}")

    if standard_config is None:
        standard_config = Config(weights=weights_path, image_size=None, training=None, model_params=override_model_params)
    # print(standard_config)
    # raise NotImplementedError

    model_wrapper = info.model_class(**standard_config.model_params)  # if standard_config is not None else info.model_class() 

    # print(weights_path)
    weights_path = weights_path if weights_path is not None else standard_config.weights
    # print(weights_path)
    if weights_path:
        weights_path = Path(weights_path)
        if weights_path.exists():
            print(f"[INFO] Weights loading from {weights_path}...")
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                
            model_wrapper.load_state_dict(state_dict, strict=False)
        else:
            print(f"[WARNING] Download skip: file {weights_path} not found or directory.")

    return model_wrapper.to(device), standard_config