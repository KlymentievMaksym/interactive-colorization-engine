import sys
from pathlib import Path
import torch
import torch.nn.functional as F

repo_path = Path(__file__).resolve().parent.parent / "util_models" / "CycleGAN_pix2pix"
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

from colorization_engine.models.util_models.CycleGAN_pix2pix.models.networks import define_G

from colorization_engine.models.util_models import BaseColorizer
from colorization_engine.factory.registry import register_model

EXPECTED_NETG = ["resnet_9blocks", "resnet_6blocks", "unet_128", "unet_256"]

@register_model("pix2pix")
class Pix2pixWrapper(BaseColorizer):
    def __init__(self, ngf: int = 64, netG: str = "unet_256"):
        super().__init__()
        if netG not in EXPECTED_NETG:
            raise ValueError(f"Wrong netG, received {netG}, expected {EXPECTED_NETG}")

        self.image_size = 128 if "128" in netG else 256

        self.netG = define_G(
            input_nc=4,
            output_nc=2,
            ngf=ngf,
            netG=netG
        )

    def forward(self, l_norm: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        """
        l_norm: [B, 1, H, W]
        hints: [B, 3, H, W] (ab_norm + mask)
        """
        batch_size, _, height, width = l_norm.shape
        device = l_norm.device

        if hints is None:
            hints = torch.zeros((batch_size, 3, height, width), device=device)

        x_in = torch.cat([l_norm, hints], dim=1)

        if height != self.image_size or width != self.image_size:
            x_in = F.interpolate(x_in, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

        out = self.netG(x_in)

        if height != self.image_size or width != self.image_size:
            out = F.interpolate(out, size=(height, width), mode='bilinear', align_corners=False)

        return out