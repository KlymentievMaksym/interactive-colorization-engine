import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from colorization_engine.loss import BaseLoss
from colorization_engine.factory.registry import register_loss
import kornia


@register_loss("colorization")
class ColorizationLoss(BaseLoss):
    def __init__(self, lpips_weight: float, l1_weight: float, hints_weight: float):
        super().__init__()
        self.lpips_weight = lpips_weight
        # self.ssim_weight = ssim_weight
        # self.psnr_weight = psnr_weight
        self.l1_weight = l1_weight
        self.hints_weight = hints_weight

        self.l1_loss = nn.L1Loss()

        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        # self.lpips_metric = [LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)]
        self.lpips_metric.requires_grad_(False)
        # self.lpips_metric[0].requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, l_channel: torch.Tensor, hint_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_l1 = self.l1_loss(pred, target) * self.l1_weight

        loss_hints = torch.zeros_like(loss_l1)
        if hint_mask is not None and hint_mask.sum() > 0:
            loss_hints = F.l1_loss(pred * hint_mask, target * hint_mask, reduction='sum') / (hint_mask.sum() + 1e-8)
            loss_hints = loss_hints * self.hints_weight

        l_unnorm = (l_channel + 1.0) * 50.0
        ab_pred_unnorm = pred * 110.0
        ab_target_unnorm = target * 110.0

        lab_pred = torch.cat([l_unnorm, ab_pred_unnorm], dim=1)
        lab_target = torch.cat([l_unnorm, ab_target_unnorm], dim=1)

        rgb_pred = kornia.color.lab_to_rgb(lab_pred)
        rgb_target = kornia.color.lab_to_rgb(lab_target)

        # val_ssim   = structural_similarity_index_measure(rgb_pred, rgb_target, data_range=1.0)
        # loss_ssim  = (1.0 - val_ssim) * self.ssim_weight

        # val_psnr   = peak_signal_noise_ratio(rgb_pred, rgb_target, data_range=1.0)
        # loss_psnr  = -val_psnr * self.psnr_weight

        loss_lpips = self.lpips_metric(rgb_pred, rgb_target) * self.lpips_weight
        # loss_lpips = self.lpips_metric[0](rgb_pred, rgb_target) * self.lpips_weight

        total_loss = loss_lpips + loss_l1 + loss_hints

        return total_loss, {
            "loss_total": total_loss.detach(),
            "loss_lpips": loss_lpips.detach(),
            "loss_hints": loss_hints.detach(),
            # "loss_psnr":  loss_psnr.detach(),
            "loss_l1":  loss_l1.detach(),
            # "loss_ssim":  loss_ssim.detach(),
        }