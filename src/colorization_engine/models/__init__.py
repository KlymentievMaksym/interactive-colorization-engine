from typing import Any, Dict
from pathlib import Path

import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from colorization_engine.utils import ModelNode
from colorization_engine.training.lightning_module import LitColorizer

from colorization_engine.models.mamba import MambaWrapper
from colorization_engine.models.siggraph17 import Siggraph17Wrapper
from colorization_engine.models.eccv16 import Eccv16Wrapper
from colorization_engine.models.icolorit import IColoriTWrapper
from colorization_engine.models.ddcolor import DDColorWrapper


MODEL_REGISTRY = {
    "mamba": MambaWrapper,
    "siggraph17": Siggraph17Wrapper,
    "eccv16": Eccv16Wrapper,
    "icolorit": IColoriTWrapper,
    "ddcolor": DDColorWrapper,
}

def load_colorization_model(model_name: str, weights: str | None, model_params: Dict[str, Any] | None, device: torch.device | str | None = "cuda") -> torch.nn.Module:
    """Loads models by its name from model registry using Hydra config"""
    model_name_lower = model_name.lower()

    if model_name_lower not in MODEL_REGISTRY:
        raise ValueError(f"[ERROR] Model {model_name_lower} not found in model registry. Available: {list(MODEL_REGISTRY.keys())}")

    model_params = OmegaConf.to_container(model_params, resolve=True) if model_params else {} # type: ignore

    model_class = MODEL_REGISTRY[model_name_lower]
    model_wrapper = model_class(**model_params)  # type: ignore

    if weights:
        weights_path = to_absolute_path(weights)
        path_obj = Path(weights_path)
        
        if path_obj.exists():
            print(f"[INFO] Weights loading from {weights_path}...")
            try:
                lit_model = LitColorizer.load_from_checkpoint(
                    checkpoint_path=weights_path,
                    map_location="cpu",
                    model=model_wrapper,
                    criterion=None,
                    **model_params  # type: ignore
                )
                model_wrapper = lit_model.model
            except Exception as e:
                print(f"[WARNING] Could not load via Lightning: {e}. Falling back to manual load.")
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                clean_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        clean_state_dict[k.replace("model.", "", 1)] = v
                    elif not any(k.startswith(p) for p in ["criterion.", "val_metrics.", "test_metrics."]):
                        clean_state_dict[k] = v

                missing, unexpected = model_wrapper.load_state_dict(clean_state_dict, strict=False)
                if missing:
                    print(f"[WARNING] Some keys were missing during load: {missing[:5]}...")
                if unexpected:
                    print(f"[WARNING] Some keys were unexpected during load: {unexpected[:5]}...")
        else:
            print(f"[WARNING] Download skip: file {weights_path} not found or directory.")

    return model_wrapper.to(device)