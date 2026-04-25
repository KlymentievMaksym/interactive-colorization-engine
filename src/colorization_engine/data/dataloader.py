from torch.utils.data import DataLoader

from colorization_engine.data.dataset import PairedDataset, SingleTargetFolderDataset
from colorization_engine.data.transforms import get_transforms

from colorization_engine.utils import DataloaderConfig

def get_dataloader(config: DataloaderConfig, is_train: bool = False, num_workers: int = 4):
    paths = config.data if is_train else config.val_data
    if paths is None:
        return paths
    transform = get_transforms(image_size=config.image_size, is_train=is_train)
    length = len(paths)
    match length:
        case 1:
            dataset = SingleTargetFolderDataset(dir_root=paths[0], transform=transform)
        case 2:
            dataset = PairedDataset(dir_inputs=paths[0], dir_targets=paths[1], transform=transform)
        case _:
            raise ValueError(f"Expected 1 or 2 paths, received: {paths}")
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=is_train, num_workers=num_workers)
