import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from colorization_engine.loss import BaseLoss
from colorization_engine.factory.registry import register_loss
import kornia


@register_loss("l1")
class L1Loss(BaseLoss):
    def __init__(self, l1_weight: float, hints_weight: float):
        super().__init__()
        self.l1_weight = l1_weight
        self.hints_weight = hints_weight

        self.l1_loss = nn.L1Loss()


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

        total_loss = loss_l1 + loss_hints

        return total_loss, {
            "loss_total": total_loss.detach(),
            "loss_l1":  loss_l1.detach(),
            "loss_hints": loss_hints.detach(),
        }