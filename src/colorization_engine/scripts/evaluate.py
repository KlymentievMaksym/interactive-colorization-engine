# src/colorization_engine/scripts/evaluate.py
import torch
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from colorization_engine.evaluation.metrics import ColorizationMetrics
from colorization_engine.models import load_colorization_model
from colorization_engine.data import get_dataloader
from colorization_engine.utils import EvaluateConfig

def save_result_images(l_tensor, ab_pred, ab_target, save_path, name):
    """
    Saves comparison: Input (B/W) | Prediction | Ground Truth
    Tensors are expected to be in the range [-1, 1]
    """
    def to_rgb(l, ab):
        l_phys = (l + 1.0) * 50.0
        ab_phys = ab * 110.0
        lab = torch.cat([l_phys, ab_phys], dim=0).permute(1, 2, 0).cpu().numpy()
        rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR)
        return (rgb * 255).clip(0, 255).astype(np.uint8)

    gray_ab = torch.zeros_like(ab_target)
    img_input = to_rgb(l_tensor, gray_ab)
    img_pred = to_rgb(l_tensor, ab_pred)
    img_gt = to_rgb(l_tensor, ab_target)

    comparison = np.hstack([img_input, img_pred, img_gt])
    cv2.imwrite(str(save_path / f"{name}.png"), comparison)


cs = ConfigStore.instance()
cs.store(name="evaluate_config", node=EvaluateConfig)

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def evaluate(config: EvaluateConfig):
    device_name = config.device if config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    
    print(f"[INFO] Loading model {config.model.model_name}...")
    model = load_colorization_model(config.model, device=device)
    model.eval()

    print(f"[INFO] Loading datasets for evaluation...")
    eval_paths = [to_absolute_path(p) for p in config.data.val] if config.data.val else None
    
    if not eval_paths:
        raise ValueError("[ERROR] No validation data paths provided in config.data.val")

    dataloader = get_dataloader(data_paths=eval_paths, image_size=config.image_size, is_train=False, batch_size=config.batch_size)

    metrics = ColorizationMetrics(device)

    save_dir = Path(to_absolute_path(config.output_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved to: {save_dir}")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            l_norms = batch["input"].to(device)
            ab_targets = batch["target"].to(device)

            ab_preds = model(l_norms)
            metrics.update(ab_preds, ab_targets)

            if (i * config.batch_size) < config.save_number:
                for j in range(l_norms.size(0)):
                    idx = i * config.batch_size + j
                    if idx >= config.save_number: 
                        break
                    save_result_images(l_norms[j], ab_preds[j], ab_targets[j], save_dir, f"result_{idx}")

    results = metrics.compute()
    print("\n" + "="*30)
    print(f"[INFO] Results for {config.model.model_name.upper()}:")
    for k, v in results.items():
        print(f" - [{k.upper()}]: {v:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()