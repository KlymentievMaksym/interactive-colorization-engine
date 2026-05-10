import numpy as np
import torch

from colorization_engine.utils.color_space import (
    rgb_to_lab,
    lab_to_rgb,
    normalize_l,
    normalize_ab,
    denormalize_l,
    denormalize_ab,
    kornia_lab_to_rgb,
    kornia_rgb_to_lab
)

def test_rgb_to_lab_roundtrip():
    """Ensures that converting RGB -> LAB -> RGB returns the original image (within rounding tolerance)."""
    original_rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    lab = rgb_to_lab(original_rgb)
    recovered_rgb = lab_to_rgb(lab)

    assert lab.shape == (64, 64, 3)
    assert recovered_rgb.shape == (64, 64, 3)

    diff = np.abs(original_rgb.astype(np.int32) - recovered_rgb.astype(np.int32))
    assert np.max(diff) <= 2, "RGB to LAB conversion drifted beyond acceptable rounding error."

def test_l_channel_normalization_roundtrip():
    """Tests if L channel is accurately normalized to [-1, 1] and back."""
    original_l = np.random.uniform(0, 100, (64, 64))

    norm_tensor = normalize_l(original_l)
    assert norm_tensor.shape == (1, 64, 64)
    assert norm_tensor.min() >= -1.0 and norm_tensor.max() <= 1.0

    recovered_l = denormalize_l(norm_tensor)
    assert recovered_l.shape == (64, 64)
    np.testing.assert_allclose(original_l, recovered_l, rtol=1e-5, atol=1e-5)

def test_ab_channel_normalization_roundtrip():
    """Tests if AB channels are accurately normalized to [-1, 1], permuted to CHW, and back."""
    original_ab = np.random.uniform(-110, 110, (64, 64, 2))

    norm_tensor = normalize_ab(original_ab)
    assert norm_tensor.shape == (2, 64, 64) # HWC -> CHW
    assert norm_tensor.min() >= -1.0 and norm_tensor.max() <= 1.0

    recovered_ab = denormalize_ab(norm_tensor)
    assert recovered_ab.shape == (64, 64, 2) # CHW -> HWC
    np.testing.assert_allclose(original_ab, recovered_ab, rtol=1e-5, atol=1e-5)

def test_kornia_rgb_to_lab_shapes_and_ranges():
    """Verifies that an RGB tensor is correctly converted to normalized L and AB tensors."""
    batch_size, h, w = 2, 32, 32
    rgb = torch.rand(batch_size, 3, h, w)

    l_norm, ab_norm = kornia_rgb_to_lab(rgb)

    assert l_norm.shape == (batch_size, 1, h, w), f"Expected L shape (B, 1, H, W), got {l_norm.shape}"
    assert ab_norm.shape == (batch_size, 2, h, w), f"Expected AB shape (B, 2, H, W), got {ab_norm.shape}"

    assert l_norm.min() >= -1.0 and l_norm.max() <= 1.0
    assert ab_norm.min() >= -1.0 and ab_norm.max() <= 1.0

def test_kornia_lab_to_rgb_shapes():
    """Verifies that normalized L and AB tensors are correctly combined back into an RGB tensor."""
    batch_size, h, w = 2, 32, 32
    l_norm = torch.rand(batch_size, 1, h, w) * 2.0 - 1.0
    ab_norm = torch.rand(batch_size, 2, h, w) * 2.0 - 1.0

    rgb = kornia_lab_to_rgb(l_norm, ab_norm)

    assert rgb.shape == (batch_size, 3, h, w), f"Expected RGB shape (B, 3, H, W), got {rgb.shape}"
    assert rgb.dtype == torch.float32

def test_kornia_roundtrip():
    """Verifies that the RGB -> LAB -> RGB conversion returns the original image."""
    batch_size, h, w = 2, 64, 64
    original_rgb = torch.rand(batch_size, 3, h, w)

    l_norm, ab_norm = kornia_rgb_to_lab(original_rgb)
    recovered_rgb = kornia_lab_to_rgb(l_norm, ab_norm)

    torch.testing.assert_close(recovered_rgb, original_rgb, rtol=1e-3, atol=1e-2)