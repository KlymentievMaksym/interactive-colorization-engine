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
        l_channel: Tensor форми [B, 1, H, W], нормований у [0, 100] (L канал LAB)
        hints: Tensor форми [B, 3, H, W] (маска + a/b значення) або None
        Returns: Tensor форми [B, 2, H, W], нормований у [-128, 127] (a/b канали)
        """
        pass

    # @abstractmethod
    # def sample_diversity(self, l_channel: torch.Tensor, num_samples: int = 5) -> torch.Tensor:
    #     """
    #     Генерує багатоваріантні результати (для дифузій або CVAE).
    #     Returns: Tensor форми [B, num_samples, 2, H, W]
    #     """
    #     pass