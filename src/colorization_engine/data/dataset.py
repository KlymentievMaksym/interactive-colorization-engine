from albumentations import Compose
import cv2
import numpy as np
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def _basic_prepare(path: str | Path, color_to: int):
    """
    Reads image from path, converts it to needed color system and returns image
    """
    path_str = str(path)
    _img = cv2.imread(path_str, cv2.IMREAD_COLOR)

    if _img is None:
        raise ValueError(f"[ERROR] Can't read: {path_str}")

    _img = cv2.cvtColor(_img, color_to)
    return _img

def __rgb_to_lab(image_rgb: np.ndarray):
    """Returns cv2 LAB format in float32 from cv2 RGB uint8"""
    img_float = image_rgb.astype(np.float32) / 255.0
    return cv2.cvtColor(img_float, cv2.COLOR_RGB2LAB)

def _l_to_l_norm(l: np.ndarray):
    return torch.from_numpy(l / 50.0 - 1.0).unsqueeze(0).float()

def _ab_to_ab_norm(ab: np.ndarray):
    return torch.from_numpy(ab / 110.0).permute(2, 0, 1).float()

def _rgb_to_lab_norm(image_rgb: np.ndarray):
    img_lab = __rgb_to_lab(image_rgb)
    return _l_to_l_norm(img_lab[:, :, 0]), _ab_to_ab_norm(img_lab[:, :, 1:])

def _rgb_to_l_norm(image_rgb: np.ndarray):
    img_lab = __rgb_to_lab(image_rgb)
    return _l_to_l_norm(img_lab[:, :, 0])

def _rgb_to_ab_norm(image_rgb: np.ndarray):
    img_lab = __rgb_to_lab(image_rgb)
    return _ab_to_ab_norm(img_lab[:, :, 1:])


def _apply_transform(transform: Compose | None, input: np.ndarray, target: np.ndarray | None = None):
    """
    Applies tranform if exists and returns dict of input and target
    """
    if transform is None:
        return {"input": input, "target": target}

    transformed = transform(image=input, target=target)
    
    return {"input": transformed['image'], "target": target if target is None else transformed['target']}

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

        tensor_dict = _apply_transform(self.transform, img_input, img_target)

        tensor_input = _rgb_to_l_norm(tensor_dict["input"])
        tensor_target = _rgb_to_ab_norm(tensor_dict["target"])

        return {"input": tensor_input, "target": tensor_target}


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

        tensor_dict = _apply_transform(self.transform, img_target)

        img_input = tensor_dict["input"]
        intar = _rgb_to_lab_norm(img_input)

        return {"input": intar[0], "target": intar[1]}