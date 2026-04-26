import torch
import torch.nn as nn

from colorization_engine.models.util_models import BaseColorizer
from colorization_engine.models.util_models.colorizers.colorizers import siggraph17 

class Siggraph17Wrapper(BaseColorizer):
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__()

        self.model = siggraph17(pretrained=pretrained)
        
        # self.model.eval()
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, l_norm: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        """
        l_norm: Tensor [B, 1, H, W] in range [-1, 1]
        hints: Tensor [B, 3, H, W]
        """
        # [-1, 1] -> [0, 100]
        l_zhang = (l_norm + 1.0) * 50.0

        if hints is None:
            ab_raw = self.model(l_zhang)
        else:
            raise NotImplementedError("Interactive")

        # ~[-110, 110] -> [-1, 1]
        ab_pred = ab_raw / 110.0

        return ab_pred