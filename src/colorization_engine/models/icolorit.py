import sys
from pathlib import Path
import torch
import torch.nn as nn

EXTERNAL_DIR = Path(__file__).resolve().parent / "util_models" / "iColoriT"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from modeling import IColoriT 

class IColoriTWrapper(nn.Module):
    def __init__(self, weights_name: str = "icolorit_base_4ch_patch16_224"):
        pass

    def forward(self, l_norm: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        pass