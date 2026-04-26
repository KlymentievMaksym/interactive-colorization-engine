import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

EXTERNAL_DIR = Path(__file__).resolve().parent / "util_models" / "DDColor"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from basicsr.archs.ddcolor_arch import DDColor

class DDColorWrapper(nn.Module):
    def __init__(self, weights_name="ddcolor_paper_tiny.pth"):
        pass

    def forward(self, l_norm: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        pass