import torch
from albumentations import Compose
import cv2
import numpy as np

from colorization_engine.utils.saliency import FastSaliencySampler
from colorization_engine.utils.patches import get_gaussian_patch_circle as get_gaussian_patch

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

def _receive_hints(ab_tensor: torch.Tensor, l_tensor: torch.Tensor, min_hint_size: int = 2, max_hint_size: int = 16, num_hints_val: int = 3, patch_size_val: int = 15, training: bool = True) -> torch.Tensor:
    _, h, w = ab_tensor.shape
    device = ab_tensor.device
    mask = torch.zeros((1, h, w), dtype=torch.float32, device=device)

    if training:
        dist = torch.distributions.Geometric(probs=torch.tensor([0.2]))
        num_hints = int(dist.sample().item())

        points = saliency_sampler.sample_points(l_tensor, num_hints * 3) if num_hints > 0 else []

        if points and num_hints > 0:
            placed_points = []
            min_dist_sq = (max_hint_size * 1.5) ** 2 

            for y, x in points:
                too_close = any((y - py)**2 + (x - px)**2 < min_dist_sq for py, px in placed_points)
                if too_close:
                    continue

                placed_points.append((y, x))

                pad = torch.randint(min_hint_size, max_hint_size, (1,), device=device).item()
                patch_size = pad * 2 + 1
                base_patch = get_gaussian_patch(patch_size, device=device)

                intensity = torch.rand(1, device=device).item() * 0.8 + 0.2
                patch = base_patch * intensity

                y1, y2 = max(0, y - pad), min(h, y + pad + 1)
                x1, x2 = max(0, x - pad), min(w, x + pad + 1)

                py1, py2 = max(0, pad - y), patch_size - max(0, y + pad + 1 - h)
                px1, px2 = max(0, pad - x), patch_size - max(0, x + pad + 1 - w)

                mask[0, y1:y2, x1:x2] = torch.maximum(mask[0, y1:y2, x1:x2], patch[py1:py2, px1:px2])

                if len(placed_points) >= num_hints:
                    break

    else:
        pad = patch_size_val // 2
        inhibition_radius = patch_size_val * 2 

        base_patch = get_gaussian_patch(size=patch_size_val, device=device)

        mag = ab_tensor.pow(2).sum(dim=0) 

        for _ in range(num_hints_val):
            idx = mag.argmax().item()
            y = idx // w
            x = idx % w

            y1, y2 = max(0, y - pad), min(h, y + pad + 1)
            x1, x2 = max(0, x - pad), min(w, x + pad + 1)
            
            py1, py2 = max(0, pad - y), patch_size_val - max(0, y + pad + 1 - h)
            px1, px2 = max(0, pad - x), patch_size_val - max(0, x + pad + 1 - w)

            mask[0, y1:y2, x1:x2] = torch.maximum(mask[0, y1:y2, x1:x2], base_patch[py1:py2, px1:px2])

            iy1, iy2 = max(0, y - inhibition_radius), min(h, y + inhibition_radius)
            ix1, ix2 = max(0, x - inhibition_radius), min(w, x + inhibition_radius)
            mag[iy1:iy2, ix1:ix2] = -1.0

    hint_ab = ab_tensor * mask
    hints = torch.cat([hint_ab, mask], dim=0)
    return hints