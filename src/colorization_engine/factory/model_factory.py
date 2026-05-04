from typing import Any, Dict
from os.path import exists
import logging

import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
# from omegaconf import OmegaConf

from colorization_engine.training.lightning_module import LitColorizer
from colorization_engine.factory.registry import MODEL_REGISTRY
import colorization_engine.models


LOGGER = logging.getLogger("ModelFactory")

def build_model(model_name: str, model_params: Dict[str, Any] | None) -> nn.Module:
    model_name_lower = model_name.lower()
    if model_name_lower not in MODEL_REGISTRY:
        raise NameError(f"Model {model_name_lower} not found in model registry. Expected {list(MODEL_REGISTRY.keys())}")
    # model_params = OmegaConf.to_container(model_params, resolve=True) if model_params else {}
    model_params = model_params if model_params else {}
    return MODEL_REGISTRY[model_name_lower](**model_params)

def load_from_lightning_checkpoint(model: nn.Module, path: str) -> nn.Module:
    lit_model = LitColorizer.load_from_checkpoint(checkpoint_path=path, map_location="cpu", model=model, criterion=None)
    return lit_model.model

def extract_state_dict(ckpt: str):
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            clean_state_dict[k.replace("model.", "", 1)] = v
        elif not any(k.startswith(p) for p in ["criterion.", "val_metrics.", "test_metrics."]):
            clean_state_dict[k] = v
    return clean_state_dict

def apply_state_dict(state_dict: dict, model: nn.Module):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.warning("Some keys were missing during load: %s...", missing[:5])
    if unexpected:
        LOGGER.warning("Some keys were unexpected during load: %s...", unexpected[:5])
    return model


def load_model_weights(model: nn.Module, path: str | None) -> nn.Module:
    if not path:
        return model

    path = to_absolute_path(path)

    if not exists(path):
        LOGGER.warning("Download skip: file %s not found or directory.", path)
        return model

    print(f"[INFO] Weights loading from {path}...")
    try:
        return load_from_lightning_checkpoint(model, path)
    except Exception as e:
        LOGGER.warning("[WARNING] Could not load via Lightning: %s. Falling back to manual load.", e)
        state_dict = extract_state_dict(path)
        return apply_state_dict(state_dict=state_dict, model=model)

def build_model_pipeline(model_name: str, weights_path: str | None, model_params: Dict[str, Any] | None, device: torch.device | str = "cuda") -> torch.nn.Module:
    """Builds models using its name, path and model_params, while sending it to device"""
    model = build_model(model_name=model_name, model_params=model_params)
    model = load_model_weights(model=model, path=weights_path)
    try:
        return model.to(device)
    except Exception as e:
        raise ValueError(f"Device {device} not found.") from e
