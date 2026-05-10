# colorization-engine/src/colorization_engine/data/datasets/paired.py
import cv2
from pathlib import Path
from torch.utils.data import Dataset

from colorization_engine.data.datasets.preparements import VALID_EXTENSIONS, _basic_prepare, _apply_transform, _receive_hints
from colorization_engine.utils.color_space import rgb_to_lab, normalize_l, normalize_ab


class PairedDataset(Dataset):
    """
    Dataset for already existing input and targets
    """
    def __init__(self, dir_inputs: str, dir_targets: str, min_hint_size: int = 2, max_hint_size: int = 16, num_hints_val: int = 3, patch_size_val: int = 15, transform = None, training: bool = False):
        self.dir_inputs = Path(dir_inputs)
        self.dir_targets = Path(dir_targets)

        self.min_hint_size = min_hint_size
        self.max_hint_size = max_hint_size
        self.num_hints_val = num_hints_val
        self.patch_size_val = patch_size_val

        self.transform = transform
        self.training = training

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

        transform_dict = _apply_transform(self.transform, input=img_input, target=img_target)

        l_tensor = normalize_l(rgb_to_lab(transform_dict["input"])[:, :, 0])
        ab_tensor = normalize_ab(rgb_to_lab(transform_dict["target"])[:, :, 1:3])

        hints_tensor = _receive_hints(ab_tensor, l_tensor, min_hint_size=self.min_hint_size, max_hint_size=self.max_hint_size, num_hints_val=self.num_hints_val, patch_size_val=self.patch_size_val, training=self.training)

        return {
            "input": l_tensor,    # [1, 256, 256]
            "hints": hints_tensor,  # [3, 256, 256] -> (a, b, mask)
            "target": ab_tensor   # [2, 256, 256]
        }
