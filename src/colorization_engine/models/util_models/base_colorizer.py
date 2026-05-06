from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseColorizer(nn.Module, ABC):
    """
    Абстрактний базовий клас для всіх моделей колоризації.
    Забезпечує єдиний інтерфейс для інференсу та валідації.
    """
    @abstractmethod
    def forward(self, l_channel: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        """
        l_channel: Tensor форми [B, 1, H, W] [-1, 1] (L)
        hints: Tensor форми [B, 3, H, W] (ab + mask) or None
        Returns: Tensor форми [B, 2, H, W] [-1, 1] (ab)
        """
        pass

    # @abstractmethod
    # def sample_diversity(self, l_channel: torch.Tensor, num_samples: int = 5) -> torch.Tensor:
    #     """
    #     Генерує багатоваріантні результати (для дифузій або CVAE).
    #     Returns: Tensor форми [B, num_samples, 2, H, W]
    #     """
    #     pass