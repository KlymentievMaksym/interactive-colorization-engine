import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from colorization_engine.loss import BaseLoss
from colorization_engine.factory.registry import register_loss
import kornia


@register_loss("colorization")
class ColorizationLoss(BaseLoss):
    def __init__(self, lpips_weight: float, l1_weight: float, hints_weight: float):
        super().__init__()
        self.lpips_weight = lpips_weight
        self.l1_weight = l1_weight
        self.hints_weight = hints_weight

        self.l1_loss = nn.L1Loss()

        _lpips_module = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        self.lpips_net = _lpips_module.net
        self.lpips_net.eval()
        self.lpips_net.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, l_channel: torch.Tensor, hint_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_l1 = self.l1_loss(pred, target) * self.l1_weight

        loss_hints = torch.zeros_like(loss_l1)
        if hint_mask is not None and hint_mask.any():
            loss_hints_raw = F.l1_loss(pred * hint_mask, target * hint_mask, reduction='none')

            loss_hints_per_image = loss_hints_raw.sum(dim=[1, 2, 3])
            hint_pixels_per_image = hint_mask.sum(dim=[1, 2, 3]).clamp(min=1e-8)

            loss_hints_avg = loss_hints_per_image / hint_pixels_per_image
            loss_hints = loss_hints_avg.mean() * self.hints_weight

        l_unnorm = (l_channel + 1.0) * 50.0
        ab_pred_unnorm = pred * 110.0
        ab_target_unnorm = target * 110.0

        lab_pred = torch.cat([l_unnorm, ab_pred_unnorm], dim=1)
        lab_target = torch.cat([l_unnorm, ab_target_unnorm], dim=1)

        rgb_pred = kornia.color.lab_to_rgb(lab_pred).clamp(0.0, 1.0)
        rgb_target = kornia.color.lab_to_rgb(lab_target).clamp(0.0, 1.0)

        rgb_pred_norm = rgb_pred * 2.0 - 1.0
        rgb_target_norm = rgb_target * 2.0 - 1.0

        loss_lpips = self.lpips_net(rgb_pred_norm, rgb_target_norm).mean() * self.lpips_weight

        total_loss = loss_l1 + loss_hints + loss_lpips

        return total_loss, {
            "loss_total": total_loss.detach(),
            "loss_l1":  loss_l1.detach(),
            "loss_hints": loss_hints.detach(),
            "loss_lpips": loss_lpips.detach(),
        }