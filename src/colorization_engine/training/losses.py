# import torch.nn as nn
# import torchvision.models as models

# class ColorizationLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.loss_fn = nn.L1Loss()

#     def forward(self, pred, target):
#         """
#         pred: (B, 3, H, W) - тензор згенерованої картинки
#         target: (B, 3, H, W) - тензор оригінальної картинки
#         """
#         loss = self.loss_fn(pred, target)

#         return loss, {"l1_loss": loss.item()}

import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorizationLoss(nn.Module):
    def __init__(self, lambda_smooth: float = 1.0, lambda_cosine: float = 2.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        
        # Наскільки сильно штрафувати за неправильний ВІДТІНОК (напрямок кольору)
        self.lambda_cosine = lambda_cosine 
        
        # Smooth L1 працює краще за звичайний L1
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, pred_ab, target_ab, l_channel=None):
        # 1. Smooth L1 Loss (Базова помилка насиченості)
        loss_smooth = self.smooth_l1(pred_ab, target_ab)
        
        # 2. COSINE SIMILARITY LOSS (Hue Loss - Вчимо ВІДТІНКИ)
        # Розгортаємо тензори, щоб порівнювати (a,b) як вектори для кожного пікселя
        pred_flat = pred_ab.permute(0, 2, 3, 1).reshape(-1, 2)     # (B*H*W, 2)
        target_flat = target_ab.permute(0, 2, 3, 1).reshape(-1, 2) # (B*H*W, 2)
        
        # Щоб уникнути ділення на нуль для ідеально сірих пікселів (0,0)
        eps = 1e-8
        
        # Рахуємо косинусну відстань
        cosine_sim = F.cosine_similarity(pred_flat + eps, target_flat + eps, dim=1)
        loss_cosine = (1.0 - cosine_sim).mean()
        
        # 3. Фінальна сума (Тільки кольори, ніякої зайвої структури)
        total_loss = (self.lambda_smooth * loss_smooth) + (self.lambda_cosine * loss_cosine)
        
        return total_loss, {
            "smooth_l1": loss_smooth.item(), 
            "cosine_hue": loss_cosine.item()
        }