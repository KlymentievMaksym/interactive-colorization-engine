from typing import Any, Dict

import torch.nn as nn

from colorization_engine.factory.registry import LOSS_REGISTRY
import colorization_engine.loss


def build_loss(loss_name: str, loss_params: Dict[str, Any] | None) -> nn.Module:
    loss_name_lower = loss_name.lower()
    if loss_name_lower not in LOSS_REGISTRY:
        raise NameError(f"Loss {loss_name_lower} not found in loss registry. Expected {list(LOSS_REGISTRY.keys())}")
    loss_params = loss_params if loss_params else {}
    return LOSS_REGISTRY[loss_name](**loss_params)