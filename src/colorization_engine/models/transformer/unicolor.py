import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch
import numpy as np
import kornia

sys.modules['ImageMatch'] = MagicMock()
sys.modules['ImageMatch.warp'] = MagicMock()
sys.modules['clip'] = MagicMock()

repo_path = Path(__file__).resolve().parent.parent / "util_models" / "UniColor"
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

sample_path = repo_path / "sample"
if str(sample_path) not in sys.path:
    sys.path.insert(0, str(sample_path))

framework_path = repo_path / "framework"
if str(framework_path) not in sys.path:
    sys.path.insert(0, str(framework_path))

from colorizer import Colorizer
from colorization_engine.models.util_models import BaseColorizer
from colorization_engine.factory.registry import register_model

@register_model("unicolor")
class UnicolorWrapper(BaseColorizer):
    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        self.dev = str(device)
        self.model = Colorizer(ckpt_path, self.dev, [256, 256], load_clip=False, load_warper=False)

    def _batch_tensor_to_points(self, hints_tensor: torch.Tensor, l_tensor: torch.Tensor):
        """
        Векторизована генерація точок для всього батчу повністю на GPU!
        hints_tensor: [B, 3, H, W]
        l_tensor: [B, 1, H, W]
        Повертає: список списків словників (по одному для кожного фото в батчі).
        """
        batch_size = hints_tensor.shape[0]
        batch_points = [[] for _ in range(batch_size)]

        masks = hints_tensor[:, 2:3, :, :]
        indices = torch.nonzero(masks)  # [N, 4] -> (b, c, y, x)

        if indices.numel() == 0:
            return batch_points

        b_idx, y_idx, x_idx = indices[:, 0], indices[:, 2], indices[:, 3]

        l_vals = l_tensor[b_idx, 0, y_idx, x_idx]
        a_vals = hints_tensor[b_idx, 0, y_idx, x_idx]
        b_vals = hints_tensor[b_idx, 1, y_idx, x_idx]

        l_unnorm = (l_vals + 1.0) * 50.0
        a_unnorm = a_vals * 110.0
        b_unnorm = b_vals * 110.0

        # [N, 3] -> [N, 3, 1, 1]
        lab_pixels = torch.stack([l_unnorm, a_unnorm, b_unnorm], dim=1).unsqueeze(-1).unsqueeze(-1)
        rgb_pixels = kornia.color.lab_to_rgb(lab_pixels)  # [N, 3, 1, 1] [0, 1]

        rgb_pixels_np = (rgb_pixels.squeeze(-1).squeeze(-1).cpu().numpy() * 255.0).astype(int)
        b_idx_np = b_idx.cpu().numpy()
        y_idx_np = y_idx.cpu().numpy()
        x_idx_np = x_idx.cpu().numpy()

        for i in range(len(b_idx_np)):
            batch_num = b_idx_np[i]
            batch_points[batch_num].append({
                'index': [int(y_idx_np[i]), int(x_idx_np[i])],
                'color': rgb_pixels_np[i].tolist()
            })

        return batch_points

    def forward(self, l_norm: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = l_norm.shape[0]

        # L -> RGB [B, 3, H, W] [0, 255]
        l_img_01 = (l_norm + 1.0) / 2.0
        l_img_rgb_tensor = l_img_01.repeat(1, 3, 1, 1) * 255.0

        l_imgs_np = l_img_rgb_tensor.cpu().numpy().astype(np.uint8)
        # PyTorch [B, 3, H, W] -> NumPy [B, H, W, 3]
        l_imgs_np = np.transpose(l_imgs_np, (0, 2, 3, 1))

        batch_points = [[] for _ in range(batch_size)]
        if hints is not None:
            batch_points = self._batch_tensor_to_points(hints, l_norm)

        rgb_results_np = np.zeros_like(l_imgs_np, dtype=np.float32)
        with torch.no_grad():
            for i in range(batch_size):
                res = self.model.sample(l_imgs_np[i], batch_points[i], topk=100)
                rgb_results_np[i] = res

        rgb_res_tensor = torch.from_numpy(rgb_results_np).permute(0, 3, 1, 2).to(l_norm.device)
        rgb_res_tensor = rgb_res_tensor / 255.0

        lab_res_tensor = kornia.color.rgb_to_lab(rgb_res_tensor)

        ab_res_tensor = lab_res_tensor[:, 1:, :, :] / 110.0

        return ab_res_tensor