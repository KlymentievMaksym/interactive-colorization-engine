from torch.utils.data import DataLoader

from colorization_engine.data.dataset import PairedDataset, SingleTargetFolderDataset
from colorization_engine.data.transforms import get_transforms


def get_dataloader(data_paths: list[str] | None, image_size: int, is_train: bool = False, batch_size: int = 4, num_workers: int = 4) -> DataLoader:
    if not data_paths:
        raise ValueError("[ERROR] No paths given")

    if batch_size is None or image_size is None:
        raise ValueError("[ERROR] No batch size given")

    length = len(data_paths)

    match length:
        case 1:
            transform = get_transforms(image_size=image_size, is_train=is_train)
            dataset = SingleTargetFolderDataset(dir_root=data_paths[0], transform=transform)
        case 2:
            transform = get_transforms(image_size=image_size, is_train=is_train, additional_targets={'target': 'image'})
            dataset = PairedDataset(dir_inputs=data_paths[0], dir_targets=data_paths[1], transform=transform)
        case _:
            raise ValueError(f"[ERROR] Expected 1 or 2 paths, received {length}: {data_paths}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, pin_memory=True, drop_last=is_train)