import torch
import torch.nn as nn
from colorization_engine.models.util_models import MambaShared, BaseColorizer


class MambaWrapper2(BaseColorizer):
    def __init__(self, d_model=256, layers=6, blocks=2):
        super().__init__()
        
        # 1. ENCODER
        # Вхід: 1 канал (L - яскравість)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ) # Вихід: (B, 64, H/2, W/2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, d_model, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ) # Вихід: (B, d_model, H/4, W/4)

        # 2. BOTTLENECK (Мамба)
        self.mamba = MambaShared(d_model=d_model, layers=layers, blocks=blocks)

        # 3. DECODER
        self.dec1 = nn.Sequential(
            # М'яке білінійне розтягування (без створення нових пікселів-галюцинацій)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # Звичайна згортка, яка просто "згладжує" розтягнуте
            nn.Conv2d(d_model, 64, kernel_size=3, padding=1),
            nn.ReLU()
        ) 

        # 4. FINAL LAYER
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # Тут вхід 128 (бо 64 з dec1 + 64 з enc1 через Skip Connection)
            nn.Conv2d(128, 2, kernel_size=3, padding=1),
            nn.Tanh() 
        )

    def forward(self, l_channel):
        """
        l_channel: (B, 1, H, W) - чорно-біла картинка
        """
        # --- ЕНКОДЕР ---
        feat1 = self.enc1(l_channel) # Зберігаємо для Skip Connection
        feat2 = self.enc2(feat1)
        
        # --- MAMBA ---
        b, c, h, w = feat2.shape
        x_flat = feat2.view(b, c, h * w).permute(0, 2, 1).contiguous()
        x_mamba = self.mamba(x_flat)
        x_res = x_mamba.permute(0, 2, 1).contiguous().view(b, c, h, w)
        
        # --- ДЕКОДЕР ---
        up1 = self.dec1(x_res)
        
        # U-Net магія: склеюємо поточні ознаки зі старими (з енкодера)
        # dim=1 означає склеювання по каналах: (B, 64, H, W) + (B, 64, H, W) -> (B, 128, H, W)
        concat_feat = torch.cat([up1, feat1], dim=1) 
        
        # Фінальний шар генерує лише два канали (кольори A та B)
        ab_channels = self.dec2(concat_feat)
        
        return ab_channels