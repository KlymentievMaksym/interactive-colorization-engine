import cv2
import numpy as np
import kornia
import torch
from einops import rearrange

def rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """Converts an RGB image (uint8, [0, 255]) to LAB (float32, L:[0, 100], ab:[-127, 127])."""
    img_float = image_rgb.astype(np.float32) / 255.0
    return cv2.cvtColor(img_float, cv2.COLOR_RGB2LAB)

def lab_to_rgb(image_lab: np.ndarray) -> np.ndarray:
    """Converts a LAB image (float32, L:[0, 100], ab:[-127, 127]) back to RGB (uint8, [0, 255])."""
    img_rgb = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
    return (img_rgb * 255.0).clip(0, 255).astype(np.uint8)

def normalize_l(l_channel: np.ndarray) -> torch.Tensor:
    """Normalizes L channel [0, 100] to [-1, 1] tensor."""
    l_tensor = torch.from_numpy(l_channel / 50.0 - 1.0).float()
    return rearrange(l_tensor, "... -> 1 ...") if l_tensor.ndim == 2 else rearrange(l_tensor, "h w 1 -> 1 h w")

def normalize_ab(ab_channels: np.ndarray) -> torch.Tensor:
    """Normalizes AB channels [-110, 110] to [-1, 1] tensor."""
    ab_tensor = torch.from_numpy(ab_channels / 110.0).float()
    return rearrange(ab_tensor, "h w c -> c h w")

def denormalize_l(l_tensor: torch.Tensor) -> np.ndarray:
    """[-1, 1] Tensor -> [0, 100] NumPy"""
    l_np = rearrange(l_tensor.detach().cpu(), "1 h w -> h w").clamp(-1, 1).numpy()
    return (l_np + 1.0) * 50.0

def denormalize_ab(ab_tensor: torch.Tensor) -> np.ndarray:
    """[-1, 1] Tensor -> [-110, 110] NumPy"""
    ab_np = rearrange(ab_tensor.detach().cpu(), "c h w -> h w c").clamp(-1, 1).numpy()
    return ab_np * 110.0

def kornia_lab_to_rgb(l_channel: torch.Tensor, ab_channels: torch.Tensor) -> torch.Tensor:
    """Convert L [-1, 1] and AB [-1, 1] tensors to RGB [0, 1]"""
    l_unnorm = (l_channel + 1.0) * 50.0
    ab_unnorm = ab_channels * 110.0

    lab = torch.cat([l_unnorm, ab_unnorm], dim=1)
    return kornia.color.lab_to_rgb(lab)

def kornia_rgb_to_lab(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert RGB [0, 1] to L [-1, 1] and AB [-1, 1] tensors"""
    lab =  kornia.color.rgb_to_lab(rgb)
    l_channel, ab_channels = lab[:, 0:1], lab[:, 1:3]

    l_norm = l_channel / 50.0 - 1.0
    ab_norm = (ab_channels / 110.0).clamp(-1, 1)

    return l_norm, ab_norm