import torch.nn as nn

from colorization_engine.models.util_models import MambaShared, BaseColorizer


class MambaWrapper(BaseColorizer):
    def __init__(self, d_model=256, layers=6, blocks=2):
        super().__init__()

        # ENCODER: (B, 3, 256, 256) -> (B, d_model, 16, 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.model = MambaShared(d_model=d_model, layers=layers, blocks=blocks)

        # DECODER: (B, d_model, 16, 16) -> (B, 3, 256, 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.encoder(x)

        # (B, C, H, W) -> (B, H*W, C)
        b, c, h, w = features.shape
        x_flat = features.view(b, c, h * w).permute(0, 2, 1).contiguous()

        x_mamba = self.model(x_flat)

        # (B, H*W, C) -> (B, C, h, w)
        x_res = x_mamba.permute(0, 2, 1).contiguous().view(b, c, h, w)

        return self.decoder(x_res) + x