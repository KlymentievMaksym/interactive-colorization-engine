# colorization-engine/src/colorization_engine/data/datasets/single.py
import cv2
from pathlib import Path
from torch.utils.data import Dataset

from colorization_engine.data.datasets.preparements import VALID_EXTENSIONS, _basic_prepare, _apply_transform, _receive_hints
from colorization_engine.utils.color_space import rgb_to_lab, normalize_l, normalize_ab


class SingleTargetFolderDataset(Dataset):
    """
    Dataset for already existing only targets
    """
    def __init__(self, dir_root, transform=None):
        self.dir_root = Path(dir_root)
        self.transform = transform
        
        self.images = [
            f for f in self.dir_root.rglob("*")
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ]

        if not len(self):
            raise RuntimeError(f"No images in {self.dir_root}")

        self.images.sort(key=lambda x: str(x))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]

        image_target = _basic_prepare(path, cv2.COLOR_BGR2RGB)

        transform_dict = _apply_transform(self.transform, input=image_target)

        image_input = transform_dict["input"]

        image_lab = rgb_to_lab(image_input)
        l_tensor = normalize_l(image_lab[:, :, 0])
        ab_tensor = normalize_ab(image_lab[:, :, 1:3])

        hints_tensor = _receive_hints(ab_tensor, l_tensor)

        return {
            "input": l_tensor,    # [1, 256, 256]
            "hints": hints_tensor,  # [3, 256, 256] -> (a, b, mask)
            "target": ab_tensor   # [2, 256, 256]
        }