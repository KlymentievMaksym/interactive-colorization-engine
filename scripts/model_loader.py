import torch
from pathlib import Path

from colorization_engine.models import MODEL_REGISTRY

def load_colorization_model(model_name: str, device: torch.device, weights: str | None = None, **override_params) -> torch.nn.Module:
    """Loads models by its name from model registry"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"[ERROR] Model {model_name} not found in model registry. Available: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[model_name]
    weights_path = Path(weights) if weights else Path(config["weights_path"])

    if not weights_path.exists():
        raise FileNotFoundError(f"[ERROR] Weights not found: {weights_path}")

    model_wrapper = config["class"](**config["params"], **override_params)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model_wrapper.load_state_dict(state_dict)

    model_wrapper = model_wrapper.to(device)

    return model_wrapper