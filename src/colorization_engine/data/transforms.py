import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(image_size: int = 256, is_train: bool = True, additional_targets: dict[str, str] | None = None) -> A.Compose:
    targets = additional_targets
    if is_train:
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=image_size),
                A.RandomCrop(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
            ],
            additional_targets=targets
        )
    else:
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size)
            ],
            additional_targets=targets
        )