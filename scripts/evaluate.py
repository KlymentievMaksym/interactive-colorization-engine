import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

from colorization_engine.utils.metrics import ColorizationMetrics

from scripts import load_colorization_model
from scripts.utils import Parser, EvalConfig, get_dataloader, parse_unknown_args


def save_result_images(l_tensor, ab_pred, ab_target, save_path, name):
    """
    Зберігає порівняння: Вхід (ч/б) | Прогноз | Оригінал
    Тензори очікуються в діапазоні [-1, 1]
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

def evaluate():
    known_args, unknown_args = Parser.evaluate_args()
    config = EvalConfig(**vars(known_args))
    model_params = parse_unknown_args(unknown_args)

    device = torch.device(config.device)
    model = load_colorization_model(model_name=config.model, device=device, weights=config.weights, **model_params)
    model.eval()

    print(f"[INFO] Loading datasets {', '.join(config.data)}...")
    config.val_data = config.data
    dataloader = get_dataloader(config=config, is_train=False, num_workers=4)

    metrics = ColorizationMetrics(device)

    save_dir = Path(config.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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
                    save_result_images(
                        l_norms[j], ab_preds[j], ab_targets[j], 
                        save_dir, f"result_{idx}"
                    )

    results = metrics.compute()
    print("\n" + "="*30)
    print(f"Результати для {config.model}:")
    for k, v in results.items():
        print(f" - {k.upper()}: {v:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()