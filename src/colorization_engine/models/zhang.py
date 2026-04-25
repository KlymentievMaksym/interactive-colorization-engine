import torch
import torch.nn as nn

from colorization_engine.models.util_models import BaseColorizer
from colorization_engine.models.colorizers import siggraph17 

class ZhangWrapper(BaseColorizer):
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__()
        
        # 1. Ініціалізація моделі (автоматично завантажить ваги, якщо pretrained=True)
        self.model = siggraph17(pretrained=pretrained)
        
        # 2. Оскільки це Baseline для бенчмарку, ми заморожуємо ваги 
        # і переводимо модель у режим інференсу, щоб не витрачати VRAM на градієнти
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, l_norm: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        """
        l_norm: Tensor форми [B, 1, H, W] у просторі [-1, 1]
        hints: Tensor форми [B, 3, H, W] (тут ми додамо логіку згодом)
        """
        # 1. Адаптація ВХОДУ: [-1, 1] -> [0, 100]
        l_zhang = (l_norm + 1.0) * 50.0
        
        # 2. Інференс
        if hints is None:
            ab_raw = self.model(l_zhang)
        else:
            raise NotImplementedError("Інтерактивний режим для ZhangWrapper додамо на етапі hints.")
            
        # 3. Адаптація ВИХОДУ: ~[-110, 110] -> [-1, 1]
        ab_pred = ab_raw / 110.0
        
        return ab_pred