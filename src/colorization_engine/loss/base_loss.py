from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseLoss(nn.Module, ABC):
    """
    Abstract interface class for all losses
    """
    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor, l_channel: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        pred: Tensor [B, 2, H, W] ab channel [-1, 1]
        target: Tensor [B, 2, H, W] ab channel [-1, 1]
        l_channel: Tensor [B, 1, H, W] L channel [-1, 1]
        Returns: total_loss, loss_dict
        """
        pass
