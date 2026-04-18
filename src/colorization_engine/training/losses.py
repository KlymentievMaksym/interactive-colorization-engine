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
import torchvision.models as models

class ColorizationLoss(nn.Module):
    def __init__(self, lambda_l1: float = 1.0, lambda_vgg: float = 0.1, device: str = 'cuda'):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        
        # 1. Базова L1 втрата (для загальної структури)
        self.l1_loss = nn.L1Loss()
        
        # 2. Perceptual (VGG) втрата
        # Завантажуємо VGG16 і беремо лише перші 16 шарів (до ReLU3_3)
        # Це ідеальне місце для "витягування" текстур і стилю
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].to(device)
        
        # Заморожуємо ваги VGG, бо ми не збираємося її навчати
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.vgg = vgg.eval()
        self.vgg_loss = nn.MSELoss()
        
        # Нормалізація для ImageNet (очікування VGG)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def normalize_for_vgg(self, x):
        """
        Переводить тензор з нашого формату [-1, 1] 
        у формат ImageNet, який розуміє VGG.
        """
        # Спочатку з [-1, 1] робимо [0, 1]
        x = (x + 1.0) / 2.0
        # Потім нормалізуємо під ImageNet
        x = (x - self.mean) / self.std
        return x

    def forward(self, pred, target):
        # 1. Рахуємо звичайну L1
        loss_l1 = self.l1_loss(pred, target)
        
        # 2. Рахуємо Perceptual Loss
        # Готуємо дані для VGG
        pred_vgg_ready = self.normalize_for_vgg(pred)
        target_vgg_ready = self.normalize_for_vgg(target)
        
        # Пропускаємо через VGG
        pred_features = self.vgg(pred_vgg_ready)
        target_features = self.vgg(target_vgg_ready)
        
        # Рахуємо різницю між "сприйняттям" картинок
        loss_vgg = self.vgg_loss(pred_features, target_features)
        
        # 3. Сумуємо з вагами
        total_loss = (self.lambda_l1 * loss_l1) + (self.lambda_vgg * loss_vgg)
        
        return total_loss, {"l1": loss_l1.item(), "vgg": loss_vgg.item()}