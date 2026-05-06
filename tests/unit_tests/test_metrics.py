import torch
import pytest
from colorization_engine.evaluation.metrics import ColorizationMetrics

def test_metrics_identical_images():
    metrics = ColorizationMetrics(device="cpu")

    img1 = torch.ones(1, 3, 64, 64)
    img2 = torch.ones(1, 3, 64, 64)

    metrics.update(img1, img2)
    results = metrics.compute()

    assert "psnr" in results
    assert "ssim" in results
    assert "mse" in results
    assert "mae" in results
    assert "lpips" in results
    assert results["ssim"] == pytest.approx(1.0)
    assert results["psnr"] > 40.0
    assert results["mse"] == 0.0
    assert results["mae"] == 0.0
    assert results["lpips"] == pytest.approx(0.0)

def test_metrics_different_images():
    metrics = ColorizationMetrics(device="cpu")

    img_white = torch.ones(1, 3, 64, 64)
    img_black = torch.zeros(1, 3, 64, 64)

    metrics.update(img_white, img_black)
    results = metrics.compute()

    assert results["ssim"] < 0.1
    assert results["psnr"] < 10.0
    assert results["mse"] > 0.0
    assert results["mae"] > 0.0
    assert results["lpips"] > 0.0