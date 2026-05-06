import numpy as np

from colorization_engine.utils.color_space import (
    rgb_to_lab,
    lab_to_rgb,
    normalize_l,
    normalize_ab,
    denormalize_l,
    denormalize_ab
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