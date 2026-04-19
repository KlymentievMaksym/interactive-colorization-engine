import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(image_size: int = 256, is_train: bool = True, additional_targets: dict[str, str] | None = None) -> A.Compose:
    pipeline = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size)
    ]

    if is_train:
        pipeline.extend([
            A.HorizontalFlip(p=0.5),
        ])

    # pipeline.extend([
    #     A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    #     # ToTensorV2()
    # ])

    return A.Compose(
        pipeline,
        additional_targets=additional_targets if additional_targets else {'target': 'image'}
    )