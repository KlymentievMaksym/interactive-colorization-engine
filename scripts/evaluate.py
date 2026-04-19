import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# Імпорти з твого рушія (завдяки pip install -e .)
from colorization_engine.data_loaders.dataset import SingleTargetFolderDataset
from colorization_engine.data_loaders.transforms import get_transforms
from colorization_engine.utils.metrics import ColorizationMetrics
from scripts.model_loader import load_colorization_model

def save_result_images(l_tensor, ab_pred, ab_target, save_path, name):
    """
    Зберігає порівняння: Вхід (ч/б) | Прогноз | Оригінал
    Тензори очікуються в діапазоні [-1, 1]
    """
    def to_rgb(l, ab):
        # Перехід з [-1, 1] назад у фізичні LAB
        l_phys = (l + 1.0) * 50.0
        ab_phys = ab * 110.0
        lab = torch.cat([l_phys, ab_phys], dim=0).permute(1, 2, 0).cpu().numpy()
        # Конвертація в BGR для OpenCV
        rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR)
        return (rgb * 255).clip(0, 255).astype(np.uint8)

    # Готуємо картинки
    # Для входу створюємо сірий AB (нулі)
    gray_ab = torch.zeros_like(ab_target)
    img_input = to_rgb(l_tensor, gray_ab)
    img_pred = to_rgb(l_tensor, ab_pred)
    img_gt = to_rgb(l_tensor, ab_target)

    # Склеюємо горизонтально
    comparison = np.hstack([img_input, img_pred, img_gt])
    cv2.imwrite(str(save_path / f"{name}.png"), comparison)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Використовується пристрій: {device}")

    # 1. Завантаження моделі
    # Припускаємо, що load_colorization_model повертає об'єкт BaseColorizer
    model = load_colorization_model(args.model_type, device, args.checkpoint)
    model.eval()

    # 2. Підготовка даних
    transform = get_transforms(image_size=args.size, is_train=False)
    dataset = SingleTargetFolderDataset(args.data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 3. Ініціалізація метрик
    metrics = ColorizationMetrics(device)
    
    # Створюємо папку для візуалізації
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Початок оцінки на {len(dataset)} зображеннях...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            l_norms = batch["input"].to(device)
            ab_targets = batch["target"].to(device)

            # Інференс
            ab_preds = model(l_norms)

            # Оновлюємо метрики
            metrics.update(ab_preds, ab_targets)

            # Зберігаємо перші N результатів для візуального контролю
            if i < args.save_n:
                for j in range(l_norms.size(0)):
                    idx = i * args.batch_size + j
                    if idx >= args.save_n: break
                    save_result_images(
                        l_norms[j], ab_preds[j], ab_targets[j], 
                        save_dir, f"result_{idx}"
                    )

    # 4. Фінальні результати
    results = metrics.compute()
    print("\n" + "="*30)
    print(f"Результати для {args.model_type}:")
    for k, v in results.items():
        print(f" - {k.upper()}: {v:.4f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оцінка моделей колоризації")
    parser.add_argument("--model_type", type=str, required=True, help="Тип моделі (zhang, mamba, deoldify)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Шлях до ваг .pth")
    parser.add_argument("--data_path", type=str, required=True, help="Папка з картинками COCO/Val")
    parser.add_argument("--output_dir", type=str, default="results/eval", help="Де зберегти картинки")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--save_n", type=int, default=50, help="Скільки картинок порівняння зберегти")
    
    args = parser.parse_args()
    evaluate(args)