import gc
import os
from unittest.mock import patch

import torch
import pytest

from colorization_engine.factory.model_factory import build_model_pipeline 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPU_ONLY = ["mamba"]
WEIGHTS_REQUIRED = ["ddcolor"]

SKIP_BY_DEFAULT = os.getenv("SKIP_HEAVY_TESTS", "True") == "True"

@pytest.mark.parametrize("model_name, model_params", [
    ("mamba",      {"d_model": 256, "layers": 6, "blocks": 2}),
    ("eccv16",     {"pretrained": False}),
    ("siggraph17", {"pretrained": False}),
    ("ddcolor",    {"model_size": "tiny", "weights_path": "models/ddcolor_paper_tiny.pth"}),
    ("pix2pix",    {"ngf": 64, "netG": "unet_256"}),
    # ("unicolor",    {"ckpt_path": "models/mscoco_step259999", "device": "cuda"}),
])
@pytest.mark.parametrize("batch_size, height, width", [
    (1, 64, 64),  # just quick check
    (2, 256, 256),  # check training and if several images
    (1, 224, 224)  # check not 2^n
])
@pytest.mark.parametrize("hints_present", [
    False,
    True
])
def test_all_models_forward_pass(model_name, model_params, batch_size, height, width, hints_present):
    """Verifies that every model in the registry can process a standard input."""

    if model_name in GPU_ONLY and DEVICE.type == "cpu":
        pytest.skip(f"{model_name} architecture requires a CUDA-enabled GPU to run.")

    if model_name in WEIGHTS_REQUIRED:
        with patch("torch.load", return_value={}):
            model = build_model_pipeline(model_name=model_name, weights_path=None, model_params=model_params)
    else:
        model = build_model_pipeline(model_name=model_name, weights_path=None, model_params=model_params)
    model.eval()

    l_norm = torch.randn(batch_size, 1, height, width, device=DEVICE)
    hints = torch.randn(batch_size, 3, height, width, device=DEVICE) if hints_present else None


    with torch.no_grad():
        out = model(l_norm, hints)

    assert out.shape == (batch_size, 2, height, width), f"{model_name} returned incorrect shape {out.shape}! Expected {(batch_size, 2, height, width)}"
    assert out.min() >= -1.0 and out.max() <= 1.0, f"Output values exceeded [-1, 1] range. Received {(out.min(), out.max())}"

    del model, l_norm, hints, out
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    SKIP_BY_DEFAULT,
    reason="Diffusion tests are disabled by default to save WSL resources. Set SKIP_HEAVY_TESTS=False to run."
)
@pytest.mark.parametrize("model_name, model_params", [
    ("control_color",      {"config_path": "models/cldm_v15_inpainting_infer1.yaml", "ckpt_path": "models/main_model.ckpt", "vae_path": "models/content-guided_deformable_vae.ckpt", "base_resolution": 512, "inference_steps": 2}),
    ("controlnet_recolor", {"inference_steps": 2, "prompt": None, "negative_prompt": None, "device": "cuda"}),
])
def test_diffusion_forward_pass(model_name, model_params):
    """Isolated test for diffusion to prevent WSL shutdown."""
    if DEVICE.type == "cpu":
        pytest.skip("Diffusion requires a CUDA-enabled GPU.")

    model = build_model_pipeline(model_name=model_name, weights_path=None, model_params=model_params).to(DEVICE)
    model.eval()

    # Enforce Batch Size = 1 to avoid the author's cldm.py bug
    batch_size, height, width = 1, 256, 256

    l_norm = torch.randn(batch_size, 1, height, width, device=DEVICE)
    hints = torch.randn(batch_size, 3, height, width, device=DEVICE)

    with torch.no_grad():
        out = model(l_norm, hints)
        
    assert out.shape == (batch_size, 2, height, width)

    del model, l_norm, hints, out
    gc.collect()
    torch.cuda.empty_cache()