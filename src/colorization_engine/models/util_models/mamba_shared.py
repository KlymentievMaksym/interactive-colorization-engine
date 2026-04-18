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
        idx_norm = 0
        block_length = len(self.blocks)
        for idx_block in range(block_length):
            expected_val_range = self.layers//block_length
            layers_range = range(expected_val_range if idx_block != block_length-1 else self.layers - expected_val_range)
            for _ in layers_range:
                x = self.blocks[idx_block](self.norms[idx_norm](x)) + x
                idx_norm += 1

        return x