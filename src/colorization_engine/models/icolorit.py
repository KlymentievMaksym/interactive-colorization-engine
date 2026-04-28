import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from colorization_engine.models.util_models import BaseColorizer
from colorization_engine.models.util_models.iColoriT.modeling import icolorit_base_4ch_patch16_224

class IColoriTWrapper(BaseColorizer):
    def __init__(self, weights_path: str = "models/icolorit_base_4ch_patch16_224.pth"):
        super().__init__()
        self.img_size = 224
        self.patch_size = 16
        self.model = icolorit_base_4ch_patch16_224(pretrained=False, head_mode='cnn', use_rpb=True)
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint["model"])

    def forward(self, l_norm: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, _, height, width = l_norm.shape
        device = l_norm.device

        # [batch_size, 1, height, width] -> [batch_size, 1, 224, 224]
        # if exists [batch_size, 3, height, width] -> [batch_size, 3, 224, 224]
        if height != self.img_size or width != self.img_size:
            l_resized = F.interpolate(l_norm, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            if hints is not None:
                hints_resized = F.interpolate(hints, size=(self.img_size, self.img_size), mode='nearest')
            else:
                hints_resized = None
        else:
            l_resized = l_norm
            hints_resized = hints

        # [-1, 1] -> [-0.5, 0.5]
        l_icolorit = l_resized * 0.5

        if hints_resized is None:
            ab_hints = torch.zeros((batch_size, 2, self.img_size, self.img_size), device=device)
            pixel_mask = torch.zeros((batch_size, 1, self.img_size, self.img_size), device=device)
        else:
            ab_hints = hints_resized[:, 0:2, :, :] 
            pixel_mask = hints_resized[:, 2:3, :, :]

        # [batch_size, 1, 224, 224] + [batch_size, 2, 224, 224] -> [batch_size, 3, 224, 224]
        x_lab = torch.cat([l_icolorit, ab_hints], dim=1) 

        # inv_pixel_mask = 1.0 - pixel_mask 
        patch_h, patch_w = self.img_size // self.patch_size, self.img_size // self.patch_size
    
        # [batch_size, 1, 224, 224] -> [batch_size, 1, 14, 14]
        # patch_mask_2d = F.interpolate(inv_pixel_mask, size=(patch_h, patch_w), mode='nearest')
        patch_mask_2d = F.max_pool2d(pixel_mask, kernel_size=self.patch_size, stride=self.patch_size)
        inv_patch_mask_2d = 1.0 - patch_mask_2d

        # [batch_size, 1, 14, 14] -> [batch_size, 196]
        patch_mask_flattened = inv_patch_mask_2d.view(batch_size, -1) 

        ### [batch_size, 2, 224, 224]
        # [batch_size, 196, 512]
        ab_pred_seq = self.model(x_lab, patch_mask_flattened)

        # [batch_size, 14, 14, 2, 16, 16]
        ab_pred_unpatched = ab_pred_seq.view(batch_size, patch_h, patch_w, 2, self.patch_size, self.patch_size)
        
        # Batch, Channels, Patch_H, Pixel_H, Patch_W, Pixel_W
        # [batch_size, 14, 14, 2, 16, 16] -> [batch_size, 2, 14, 16, 14, 16]
        ab_pred_img = ab_pred_unpatched.permute(0, 3, 1, 4, 2, 5).contiguous()
        
        # [batch_size, 2, 14, 16, 14, 16] -> [batch_size, 2, 224, 224]
        ab_pred_224 = ab_pred_img.view(batch_size, 2, self.img_size, self.img_size)

        # [batch_size, 2, 224, 224] -> [batch_size, 2, height, width]
        if height != self.img_size or width != self.img_size:
            ab_pred_final = F.interpolate(ab_pred_224, size=(height, width), mode='bilinear', align_corners=False)
        else:
            ab_pred_final = ab_pred_224

        return ab_pred_final