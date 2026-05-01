import torch.nn as nn
from mamba_ssm import Mamba

class MambaShared(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2, layers: int = 6, blocks: int = 2):
        super().__init__()
        self.layers = layers

        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(blocks)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])

    def forward(self, x):
        expected_val_range = self.layers // len(self.blocks)

        for i in range(self.layers):
            idx_block = min(i // expected_val_range, len(self.blocks) - 1)
            x = self.blocks[idx_block](self.norms[i](x)) + x

        return x