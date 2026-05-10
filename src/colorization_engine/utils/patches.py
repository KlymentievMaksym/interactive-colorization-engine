import torch

def get_gaussian_patch_box(size: int, sigma: float | None = None, device: str | torch.device = 'cpu') -> torch.Tensor:
    """Generate 2D Gauss kernel with max 1.0"""
    if sigma is None:
        sigma = size / 4.0
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.max()
    return g.view(-1, 1) * g.view(1, -1)

def get_gaussian_patch_circle(size: int, sigma: float | None = None, device: str | torch.device = 'cpu') -> torch.Tensor:
    """Generate 2D Radial Gauss kernel with max 1.0 and strict circular bounds"""
    if sigma is None:
        sigma = size / 4.0

    pad = size // 2
    coords = torch.arange(size, dtype=torch.float32, device=device) - pad

    Y, X = torch.meshgrid(coords, coords, indexing="ij")
    dist_sq = X**2 + Y**2

    patch = torch.exp(-dist_sq / (2 * sigma**2))
    edge_val = torch.exp(torch.tensor(-(pad**2) / (2 * sigma**2), device=device))
    patch = patch - edge_val

    patch[dist_sq > pad**2] = 0.0
    patch = patch.clamp(min=0.0)

    patch = patch / patch.max()
    
    return patch