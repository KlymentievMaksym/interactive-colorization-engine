import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddcolor.model import DDColor
from ddcolor.pipeline import build_ddcolor_model
from colorization_engine.models.util_models import BaseColorizer

class DDColorWrapper(BaseColorizer):
    def __init__(self, model_size: str = "tiny", weights_path: str = "models/ddcolor_paper_tiny.pth"):
        super().__init__()
        
        self.img_size = 512 # DDColor relies heavily on 512x512
        
        # Build the model exactly as the authors intended
        self.model = build_ddcolor_model(
            DDColor, 
            model_path=weights_path, 
            model_size=model_size,
            input_size=self.img_size
        )
        self.model.eval()

    def _l_to_rgb(self, L: torch.Tensor) -> torch.Tensor:
        """
        Fast and accurate conversion from L channel [0, 100] to neutral sRGB [0, 1].
        This perfectly simulates OpenCV's cvtColor(LAB2RGB) for grayscale images.
        """
        # Convert L to Y (Luminance)
        y = (L + 16.0) / 116.0
        mask = y > 0.2068966
        Y = torch.where(mask, torch.pow(y, 3.0), (y - 16.0 / 116.0) / 7.787)

        # For a neutral gray (a=0, b=0), R = G = B = Y (with sRGB gamma correction)
        mask_gamma = Y > 0.0031308
        # Clamp Y to prevent NaN in power function near zero
        RGB = torch.where(mask_gamma, 1.055 * torch.pow(Y.clamp(min=1e-6), 1.0 / 2.4) - 0.055, Y * 12.92)

        RGB = torch.clamp(RGB, 0.0, 1.0)
        # Repeat 3 times to create [B, 3, H, W]
        return RGB.repeat(1, 3, 1, 1)

    @torch.no_grad()
    def forward(self, l_norm: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, _, height, width = l_norm.shape

        # 1. Resize input to DDColor's required size (512x512)
        if height != self.img_size or width != self.img_size:
            l_resized = F.interpolate(l_norm, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        else:
            l_resized = l_norm

        # 2. Convert [-1, 1] L-channel to Neutral RGB [0, 1]
        L_channel = (l_resized + 1.0) * 50.0
        tensor_gray_rgb = self._l_to_rgb(L_channel)

        ab_pred_512 = self.model(tensor_gray_rgb)

        ab_pred_norm = ab_pred_512 / 110.0

        if height != self.img_size or width != self.img_size:
            ab_pred_final = F.interpolate(ab_pred_norm, size=(height, width), mode='bilinear', align_corners=False)
        else:
            ab_pred_final = ab_pred_norm

        return ab_pred_final