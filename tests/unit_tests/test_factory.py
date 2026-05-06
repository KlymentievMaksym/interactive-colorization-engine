import pytest

from colorization_engine.factory import build_model_pipeline
from colorization_engine.factory import build_loss
from colorization_engine.models import Eccv16Wrapper
from colorization_engine.loss import ColorizationLoss

def test_build_registered_model():
    """Checks creating model."""
    model_params = {"pretrained": False}
    model = build_model_pipeline(model_name="eccv16", weights_path=None, model_params=model_params)
    assert isinstance(model, Eccv16Wrapper)

def test_build_unregistered_model():
    """Checks mistyping model."""
    with pytest.raises(NameError):
        build_model_pipeline("fake_unreal_model", weights_path=None, model_params=None)

def test_build_registered_loss():
    """Checks creating loss."""
    loss_params = {"lpips_weight": 2.0, "l1_weight": 5.0, "hints_weight": 50.0}
    loss = build_loss("colorization", loss_params)
    assert isinstance(loss, ColorizationLoss)
    assert loss.lpips_weight == 2.0
    assert loss.l1_weight == 5.0

def test_build_unregistered_loss():
    """Checks mistyping loss."""
    with pytest.raises(NameError):
        build_loss("coloriztion_typo", None)