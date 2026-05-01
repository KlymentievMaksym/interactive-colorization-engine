import albumentations as A


def get_train_transforms(image_size: int = 256, additional_targets: dict[str, str] | None = None) -> A.Compose:
    """Standard spatial augmentations for training colorization models."""
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size),
            A.RandomCrop(height=image_size, width=image_size),
            A.SquareSymmetry(),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=0.05, contrast_limit=0.05, ensure_safe_range=True)
        ],
        additional_targets=additional_targets
    )

def get_val_transforms(image_size: int = 256, additional_targets: dict[str, str] | None = None) -> A.Compose:
    """Strict resizing for validation to ensure consistent metric calculation."""
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size),
            A.CenterCrop(height=image_size, width=image_size),
        ],
        additional_targets=additional_targets
    )

def get_test_transforms(image_size: int = 256, additional_targets: dict[str, str] | None = None) -> A.Compose:
    """Strict resizing for testing to ensure consistent metric calculation."""
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size),
            A.CenterCrop(height=image_size, width=image_size),
        ],
        additional_targets=additional_targets
    )