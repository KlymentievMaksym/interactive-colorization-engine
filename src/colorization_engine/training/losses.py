import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorizationLoss(nn.Module):
    def __init__(self, lambda_smooth: float = 1.0, lambda_cosine: float = 2.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        
        self.lambda_cosine = lambda_cosine 
        
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, pred_ab, target_ab):
        # 1. Smooth L1 Loss (Базова помилка насиченості)
        loss_smooth = self.smooth_l1(pred_ab, target_ab)
        
        # 2. COSINE SIMILARITY LOSS (Hue Loss - Вчимо ВІДТІНКИ)
        # Розгортаємо тензори, щоб порівнювати (a,b) як вектори для кожного пікселя
        pred_flat = pred_ab.permute(0, 2, 3, 1).reshape(-1, 2)     # (B*H*W, 2)
        target_flat = target_ab.permute(0, 2, 3, 1).reshape(-1, 2) # (B*H*W, 2)
        
        # Рахуємо косинусну відстань
        cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1, eps=1e-8)
        loss_cosine = (1.0 - cosine_sim).mean()
        
        # 3. Фінальна сума (Тільки кольори, ніякої зайвої структури)
        total_loss = (self.lambda_smooth * loss_smooth) + (self.lambda_cosine * loss_cosine)
        
        return total_loss, {
            "smooth_l1": self.lambda_smooth * loss_smooth.item(), 
            "cosine_hue": self.lambda_cosine * loss_cosine.item()
        }