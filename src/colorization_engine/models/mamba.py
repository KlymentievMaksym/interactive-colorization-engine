import torch
import torch.nn as nn
from colorization_engine.models.util_models import MambaShared, BaseColorizer

class DoubleConv(nn.Module):
    """Допоміжний блок для поглиблення рецептивного поля без втрати роздільної здатності"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class MambaWrapper(BaseColorizer):
    def __init__(self, d_model: int = 256, layers: int = 6, blocks: int = 2):
        super().__init__()
        
        # --- ENCODER ---
        # Вхід: 1 канал L + 2 канали ab (підказки) + 1 канал маски = 4 канали
        self.enc1 = DoubleConv(4, 64) 
        self.pool1 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1) # H/2

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1) # H/4
        
        self.enc3 = DoubleConv(128, d_model)
        self.pool3 = nn.Conv2d(d_model, d_model, kernel_size=4, stride=2, padding=1) # H/8

        # --- BOTTLENECK (Mamba) ---
        # Тепер Mamba працює на значно меншій роздільній здатності H/8 x W/8
        # Це експоненційно зменшує використання VRAM та дозволяє моделі бачити глобальний контекст
        self.mamba = MambaShared(d_model=d_model, layers=layers, blocks=blocks)

        # --- DECODER ---
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DoubleConv(d_model + d_model, 128) # Concat з enc3

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DoubleConv(128 + 128, 64) # Concat з enc2

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DoubleConv(64 + 64, 64) # Concat з enc1

        # --- FINAL LAYER ---
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Tanh() # Обмежуємо вихід у діапазоні [-1, 1] для каналів a та b
        )

    def forward(self, l_norm: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        """
        l_norm: (B, 1, H, W) 
        hints: (B, 3, H, W) - канали [a, b, mask], згенеровані _receive_hints
        """
        batch_size, _, height, width = l_norm.shape
        device = l_norm.device

        # Захист: якщо підказок немає (наприклад, під час повного автоматичного інференсу)
        if hints is None:
            hints = torch.zeros((batch_size, 3, height, width), device=device)
            
        # x_in матиме 4 канали, що ідеально підходить для першого шару: nn.Conv2d(4, 64, ...)
        x_in = torch.cat([l_norm, hints], dim=1)
        
        # --- ENCODER FORWARD ---
        feat1 = self.enc1(x_in)       # (B, 64, H, W)
        p1 = self.pool1(feat1)        # (B, 64, H/2, W/2)
        
        feat2 = self.enc2(p1)         # (B, 128, H/2, W/2)
        p2 = self.pool2(feat2)        # (B, 128, H/4, W/4)
        
        feat3 = self.enc3(p2)         # (B, 256, H/4, W/4)
        p3 = self.pool3(feat3)        # (B, 256, H/8, W/8)
        
        # --- MAMBA FORWARD ---
        b, c, h, w = p3.shape
        x_flat = p3.view(b, c, h * w).permute(0, 2, 1).contiguous()
        x_mamba = self.mamba(x_flat)
        x_res = x_mamba.permute(0, 2, 1).contiguous().view(b, c, h, w)
        
        # --- DECODER FORWARD ---
        d3 = self.up3(x_res)
        d3 = self.dec3(torch.cat([d3, feat3], dim=1)) # Злиття Mamba + локальний контекст H/4
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, feat2], dim=1)) # Злиття H/2
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, feat1], dim=1)) # Злиття H
        
        ab_channels = self.final_conv(d1)
        
        return ab_channels