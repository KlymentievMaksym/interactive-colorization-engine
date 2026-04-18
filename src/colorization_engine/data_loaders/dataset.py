import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def _basic_prepare(path: str | Path, color_to: int):
    """
    Reads image from path, converts it to needed color system and returns image
    """
    path_str = str(path)
    _img = cv2.imread(path_str)

    if _img is None:
        raise ValueError(f"[ERROR] Can't read: {path_str}")

    _img = cv2.cvtColor(_img, color_to)
    return _img

def _apply_transform(transform, input, target):
    """
    Applies tranform if exists and returns dict of input and target
    """
    if transform:
        transformed = transform(image=input, target=target)
        tensor_gray = transformed['image']
        tensor_color = transformed['target']
    else:
        tensor_gray, tensor_color = input, target

    return {"input": tensor_gray, "target": tensor_color}

class PairedDataset(Dataset):
    """
    Dataset for already existing input and targets
    """
    def __init__(self, dir_inputs, dir_targets, transform=None):
        self.dir_inputs = Path(dir_inputs)
        self.dir_targets = Path(dir_targets)
        self.transform = transform

        inputs = {
            f.stem: f for f in self.dir_inputs.rglob("*")
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        }
        
        self.pairs = []
        for target_path in self.dir_targets.rglob("*"):
            if target_path.is_file() and target_path.suffix.lower() in VALID_EXTENSIONS and target_path.stem in inputs:
                self.pairs.append({
                    "input": inputs[target_path.stem],
                    "target": target_path
                })

        if not len(self):
            raise RuntimeError(f"[ERROR] No images that both in {self.dir_inputs} and {self.dir_targets}")

        self.pairs.sort(key=lambda x: x["target"].stem)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        img_input = _basic_prepare(pair["input"], cv2.COLOR_BGR2RGB)
        img_target = _basic_prepare(pair["target"], cv2.COLOR_BGR2RGB)

        return _apply_transform(self.transform, img_input, img_target)


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
            raise RuntimeError(f"[ERROR] No images in {self.dir_root}")

        self.images.sort(key=lambda x: str(x))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]

        img_target = _basic_prepare(path, cv2.COLOR_BGR2RGB)
        img_input = cv2.cvtColor(img_target, cv2.COLOR_RGB2GRAY)
        img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)

        return _apply_transform(self.transform, img_input, img_target)