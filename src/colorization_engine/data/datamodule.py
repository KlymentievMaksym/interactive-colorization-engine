import logging

from typing import Protocol
from albumentations import Compose

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from colorization_engine.data.datasets.single import SingleTargetFolderDataset
from colorization_engine.data.datasets.paired import PairedDataset
from colorization_engine.data.transforms import get_train_transforms, get_val_transforms, get_test_transforms

log = logging.getLogger(__name__)

class TransformFactory(Protocol):
    def __call__(
        self,
        image_size: int,
        additional_targets: dict[str, str] | None
    ) -> Compose:
        ...

class ColorizationDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_paths: list[str | list[str]],
            val_paths: list[str | list[str]] | None = None,
            test_paths: list[str | list[str]] | None = None,
            image_size: int = 256, hint_size: int = 6,
            batch_size: int = 16, num_workers: int = 4, timeout: float = 60
        ):
        super().__init__()
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths

        self.image_size = image_size
        self.hint_size = hint_size
    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.timeout = timeout

        self.save_hyperparameters(ignore=["train_paths", "val_paths", "test_paths"])

        self.additional_targets = {"target": "image"}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _create_dataset(self, paths: list[str | list[str]] | None, get_transform: TransformFactory | None = None):
        if paths is None:
            return None

        if len(paths) == 0:
            raise ValueError("Paths list is empty")

        if get_transform is None:
            transform = None
        else:
            try:
                transform = get_transform(image_size=self.image_size, additional_targets=self.additional_targets)
            except TypeError as e:
                raise TypeError(f"Transform function must accept (image_size, additional_targets). Got: {get_transform}") from e

        dataset = []
        for path in paths:
            if isinstance(path, str):
                dataset.append(SingleTargetFolderDataset(path, hint_size=self.hint_size, transform=transform))
            elif isinstance(path, list):
                if len(path) != 2 or not all(isinstance(p, str) for p in path):
                    raise ValueError(f"Wrong path type. Paired path must be [str, str], got: {path}")
                dataset.append(PairedDataset(*path, hint_size=self.hint_size, transform=transform))
            else:
                raise TypeError(f"Wrong path type. Expected str or [str, str], got: {path}")
        return ConcatDataset(dataset)

    def setup(self, stage: str | None = None):
        match stage:
            case "fit" | None: # Lightning calls setup(None) before training - treat as "fit"
                if self.train_dataset is None:
                    self.train_dataset = self._create_dataset(
                        paths=self.train_paths,
                        get_transform=get_train_transforms
                    )
                if self.val_dataset is None:
                    self.val_dataset = self._create_dataset(
                        paths=self.val_paths,
                        get_transform=get_val_transforms
                    )
            case "validate":
                if self.val_dataset is None:
                    self.val_dataset = self._create_dataset(
                        paths=self.val_paths,
                        get_transform=get_val_transforms
                    )
            case "test":
                if self.test_dataset is None:
                    self.test_dataset = self._create_dataset(
                        paths=self.test_paths,
                        get_transform=get_test_transforms
                    )

    def _create_dataloader(self, dataset: Dataset | None, shuffle: bool = False, required: bool = False):
        if dataset is None:
            if required:
                raise RuntimeError("Dataset is not initialized")
            log.warning("Requested dataloader but dataset is None, skipping")
            return None

        return DataLoader(
            dataset, batch_size=self.batch_size, 
            shuffle=shuffle, num_workers=self.num_workers,
            pin_memory=True, persistent_workers=self.num_workers > 0,
            timeout=self.timeout
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True, required=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, shuffle=False)