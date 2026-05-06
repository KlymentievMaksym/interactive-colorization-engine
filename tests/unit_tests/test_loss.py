import torch
import pytest

from colorization_engine.factory.loss_factory import build_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOSS_CONFIGS = [
    ("colorization", {"lpips_weight": 1.0, "l1_weight": 1.0, "hints_weight": 1.0}),
    ("l1",           {"l1_weight": 1.0, "hints_weight": 1.0})
]

@pytest.mark.parametrize("loss_name, loss_params", LOSS_CONFIGS)
def test_loss_identical_tensors(loss_name, loss_params):
    """
    If the prediction and the ground truth are completely identical,
    the total loss should be basically 0 for ALL loss functions.
    """
    color_loss = build_loss(loss_name, loss_params).to(DEVICE)
    batch_size, channels, h, w = 2, 2, 64, 64

    l_channel = torch.zeros(batch_size, 1, h, w, device=DEVICE)
    hint_mask = torch.zeros(batch_size, 1, h, w, device=DEVICE)
    pred = torch.zeros(batch_size, channels, h, w, device=DEVICE)
    target = torch.zeros(batch_size, channels, h, w, device=DEVICE)

    total_loss, loss_dict = color_loss(pred, target, l_channel, hint_mask)

    assert total_loss.item() < 1e-4, f"Loss for identical images should be zero for {loss_name}!"

@pytest.mark.parametrize("loss_name, loss_params", LOSS_CONFIGS)
def test_loss_different_tensors(loss_name, loss_params):
    """
    If the model outputs complete noise, the loss should be positive.
    Verifies that hint masks and components are calculated correctly.
    """
    color_loss = build_loss(loss_name, loss_params).to(DEVICE)
    batch_size, channels, h, w = 2, 2, 64, 64

    l_channel = torch.zeros(batch_size, 1, h, w, device=DEVICE)
    hint_mask = torch.ones(batch_size, 1, h, w, device=DEVICE)
    pred = torch.ones(batch_size, channels, h, w, device=DEVICE)
    target = -torch.ones(batch_size, channels, h, w, device=DEVICE)

    total_loss, loss_dict = color_loss(pred, target, l_channel, hint_mask)

    assert total_loss.item() > 0.0, f"Total loss must be > 0 for {loss_name}"