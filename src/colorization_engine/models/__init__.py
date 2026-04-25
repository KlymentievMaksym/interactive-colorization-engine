import torch
from pathlib import Path
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from colorization_engine.utils import ModelNode

from colorization_engine.models.mamba import MambaWrapper
from colorization_engine.models.zhang import ZhangWrapper
# from colorization_engine.models.deoldify import DeOldifyWrapper

MODEL_REGISTRY = {
    "mamba": MambaWrapper,
    "zhang_siggraph17": ZhangWrapper,
    # "deoldify_artistic": DeOldifyWrapper,
}

def load_colorization_model(config: ModelNode, device: torch.device | str = "cuda") -> torch.nn.Module:
    """Loads models by its name from model registry using Hydra config"""
    model_name = config.model_name.lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"[ERROR] Model {model_name} not found in model registry. Available: {list(MODEL_REGISTRY.keys())}")

    model_params = OmegaConf.to_container(config.model_params, resolve=True) if config.model_params else {}

    model_class = MODEL_REGISTRY[model_name]
    model_wrapper = model_class(**model_params) 

    if config.weights:
        weights_path = to_absolute_path(config.weights)
        path_obj = Path(weights_path)
        
        if path_obj.exists():
            print(f"[INFO] Weights loading from {weights_path}...")
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                
            model_wrapper.load_state_dict(state_dict, strict=False)
        else:
            print(f"[WARNING] Download skip: file {weights_path} not found or directory.")

    return model_wrapper.to(device)