import pytest
from PIL import Image

from colorization_engine.data import ColorizationDataModule

@pytest.fixture
def dummy_dataset_dir(tmp_path):
    """Generate temp dir with 4 photos"""
    data_dir = tmp_path / "dummy_images"
    data_dir.mkdir()

    for i in range(4):
        img = Image.new('RGB', (300, 300), color=(i * 50, 100, 200))
        img.save(data_dir / f"test_img_{i}.jpg")

    return str(data_dir)

def test_datamodule_setup_and_dataloader(dummy_dataset_dir):
    """
    initialize -> setup -> receive batch.
    check transform (resize, to_tensor, rgb_to_lab).
    """
    dm = ColorizationDataModule(
        train_paths=[dummy_dataset_dir],
        val_paths=[dummy_dataset_dir],
        image_size=256,
        hint_size=8,
        batch_size=2,
        num_workers=0
    )

    dm.setup(stage="fit")

    train_loader = dm.train_dataloader()
    assert train_loader is not None

    batch = next(iter(train_loader))
    assert "input" in batch, "Батч має містити ключ 'input' (L канал)"
    assert "target" in batch, "Батч має містити ключ 'target' (ab канали)"

    if "hints" in batch:
        hints = batch["hints"]
        assert hints.shape == (2, 3, 256, 256)

    l_channel = batch["input"]
    ab_channels = batch["target"]

    assert l_channel.shape == (2, 1, 256, 256)
    assert ab_channels.shape == (2, 2, 256, 256)

    assert l_channel.max() <= 1.0 and l_channel.min() >= -1.0
    assert ab_channels.max() <= 1.0 and ab_channels.min() >= -1.0
