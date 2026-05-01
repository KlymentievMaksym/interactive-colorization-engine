import torch
from albumentations import Compose
import cv2
import numpy as np

from colorization_engine.utils.saliency import FastSaliencySampler

saliency_sampler = FastSaliencySampler(blur_kernel_size=15, uniform_prior=0.15)

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def _basic_prepare(path: str, color_code: int):
    """Reads image from path, converts it to needed color system and returns image"""
    _img = cv2.imread(path, cv2.IMREAD_COLOR)

    if _img is None:
        raise ValueError(f"Can't read: {path}")

    _img = cv2.cvtColor(_img, color_code)
    return _img

def _apply_transform(transform: Compose | None, input: np.ndarray, target: np.ndarray | None = None):
    """Applies tranform if exists and returns dict of input, hints and target"""
    if transform is None:
        return {"input": input, "target": target}

    transformed = transform(image=input, target=target)

    return {"input": transformed['image'], "target": target if target is None else transformed['target']}

def _receive_hints(ab_tensor: torch.Tensor, l_tensor: torch.Tensor, point_size: int = 8, training: bool = True) -> torch.Tensor:
    _, h, w = ab_tensor.shape
    device = ab_tensor.device
    mask = torch.zeros((1, h, w), dtype=torch.float32, device=device)

    if training:
        dist = torch.distributions.Geometric(probs=torch.tensor([0.2]))
        num_hints = int(dist.sample().item())

        points = saliency_sampler.sample_points(l_tensor, num_hints)
    else:
        points = []

    if points:
        y_coords = torch.tensor([p[0] for p in points], dtype=torch.long)
        x_coords = torch.tensor([p[1] for p in points], dtype=torch.long)
        
        if point_size == 1:
            mask[0, y_coords, x_coords] = 1.0
        else:
            pad = point_size // 2
            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    y_patch = torch.clamp(y_coords + dy, 0, h - 1)
                    x_patch = torch.clamp(x_coords + dx, 0, w - 1)
                    mask[0, y_patch, x_patch] = 1.0

    hint_ab = ab_tensor * mask
    hints = torch.cat([hint_ab, mask], dim=0)
    
    return hints
