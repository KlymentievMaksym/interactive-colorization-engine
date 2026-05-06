import torch
import pytest
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from colorization_engine.factory.model_factory import build_model
from colorization_engine.factory.loss_factory import build_loss
from colorization_engine.training.lightning_module import LitColorizer

class DummyDictDataset(Dataset):
    """Generates a fake dataset in the dictionary format expected by LitColorizer."""
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "input": torch.randn(1, 64, 64),
            "hints": torch.ones(3, 64, 64),
            "target": torch.randn(2, 64, 64)
        }

def test_lightning_fast_dev_run():
    """
    The ultimate integration test. Runs exactly 1 full training and validation step
    to prove the entire architecture (Model + Loss + Optimizer) is wired correctly.
    """
    model = build_model("eccv16", {"pretrained": False})
    criterion = build_loss("l1", {"l1_weight": 1.0, "hints_weight": 1.0})
    lit_model = LitColorizer(model=model, criterion=criterion, epochs=1, lr=1e-4, amount_show=1)

    train_loader = DataLoader(DummyDictDataset(), batch_size=2)
    val_loader = DataLoader(DummyDictDataset(), batch_size=2)

    trainer = pl.Trainer(
        fast_dev_run=True,  # fast_dev_run=True processes exactly 1 batch and disables logging/checkpoints
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False
    )

    try:
        trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except Exception as e:
        pytest.fail(f"PyTorch Lightning Trainer crashed! Error: {e}")