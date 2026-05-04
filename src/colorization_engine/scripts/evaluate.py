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
from colorization_engine.factory import build_model_pipeline
# ВИПРАВЛЕННЯ 1: Використовуємо існуючий DataModule замість неіснуючої функції
from colorization_engine.data import ColorizationDataModule
from colorization_engine.utils import EvaluateConfig
from colorization_engine.utils.color_space import denormalize_l, denormalize_ab, lab_to_rgb

def batch_lab_to_rgb_metrics(l_norm: torch.Tensor, ab_norm: torch.Tensor) -> torch.Tensor:
    """
    Конвертує батч [-1, 1] LAB тензорів у батч [0, 1] RGB тензорів для метрик.
    
    Args:
        l_norm (torch.Tensor): Тензор [B, 1, H, W] у діапазоні [-1.0, 1.0].
        ab_norm (torch.Tensor): Тензор [B, 2, H, W] у діапазоні [-1.0, 1.0].
        
    Returns:
        torch.Tensor: Тензор [B, 3, H, W] у діапазоні [0.0, 1.0].
    """
    device = l_norm.device
    b, _, h, w = l_norm.shape
    
    # Відв'язуємо від графа та переносимо на CPU для роботи з OpenCV
    l_np = l_norm.detach().cpu().numpy()   # [B, 1, H, W]
    ab_np = ab_norm.detach().cpu().numpy() # [B, 2, H, W]
    
    rgb_tensors = []
    
    for i in range(b):
        # 1. Денормалізація [-1, 1] -> Фізичні значення
        # Використовуємо .squeeze() для переходу від [1, H, W] до [H, W]
        l_phys = (l_np[i].squeeze(0) + 1.0) * 50.0       # [H, W], діапазон [0, 100]
        
        # Переставляємо осі для ab: [2, H, W] -> [H, W, 2]
        ab_phys = np.transpose(ab_np[i], (1, 2, 0)) * 110.0 # [H, W, 2], діапазон [-110, 110]
        
        # 2. Об'єднання в один LAB масив: [H, W, 3]
        lab_img = np.empty((h, w, 3), dtype=np.float32)
        lab_img[..., 0] = l_phys
        lab_img[..., 1:] = ab_phys
        
        # 3. Конвертація в RGB [0, 255] uint8
        # cv2.COLOR_LAB2RGB автоматично розуміє формат float32 з L[0,100] та ab[-128,127]
        rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB) 
        rgb_img = np.clip(rgb_img * 255.0, 0, 255).astype(np.uint8)
        
        # 4. Нормалізація [0, 255] -> [0.0, 1.0] та перетворення у тензор
        # numpy [H, W, C] -> torch [C, H, W]
        rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
        rgb_tensors.append(rgb_tensor)
        
    # Збираємо батч і повертаємо на оригінальний пристрій (GPU/CPU)
    return torch.stack(rgb_tensors).to(device)

def save_result_images(rgb_input: torch.Tensor, rgb_pred: torch.Tensor, rgb_target: torch.Tensor, save_path: Path, name: str):
    """Зберігає порівняння: B/W Input | Prediction | Ground Truth"""
    def to_bgr_numpy(t: torch.Tensor) -> np.ndarray:
        img = (t.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_in = to_bgr_numpy(rgb_input)
    img_gray = cv2.cvtColor(cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    
    img_p = to_bgr_numpy(rgb_pred)
    img_gt = to_bgr_numpy(rgb_target)

    comparison = np.hstack([img_gray, img_p, img_gt])
    cv2.imwrite(str(save_path / f"{name}.png"), comparison)


cs = ConfigStore.instance()
cs.store(name="evaluate_config", node=EvaluateConfig)

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def evaluate(config: EvaluateConfig):
    device = torch.device(config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    print(f"[INFO] Loading model {config.model.model_name}...")
    model = build_model_pipeline(
        model_name=config.model.model_name, 
        weights_path=config.model.weights, 
        model_params=config.model.model_params, 
        device=device
    )
    model.eval()

    # ВИПРАВЛЕННЯ 3: Коректна обробка вкладених списків для шляхів (аналогічно до train.py)
    eval_paths = [
        to_absolute_path(p) if isinstance(p, str) 
        else [to_absolute_path(p[0]), to_absolute_path(p[1])] 
        for p in config.data.val
    ] if config.data.val else None

    if not eval_paths:
        raise ValueError("[ERROR] No validation data paths provided in config.data.val")

    # Ініціалізація DataModule виключно для валідаційної вибірки
    datamodule = ColorizationDataModule(
        train_paths=[], 
        val_paths=eval_paths,
        image_size=config.image_size, 
        batch_size=config.batch_size
    )
    
    # ВИПРАВЛЕННЯ: Використовуємо stage="fit" або "validate" (залежить від вашої реалізації DataModule), 
    # або викликаємо setup() без аргументів, щоб Lightning ініціалізував усі доступні спліти.
    datamodule.setup(stage="validate") 
    
    dataloader = datamodule.val_dataloader()
    
    # Додатковий захист від порожнього даталоадера
    if dataloader is None:
        # Якщо stage="validate" не прописаний у вашому DataModule, викликаємо глобальний setup
        datamodule.setup()
        dataloader = datamodule.val_dataloader()
        
    if dataloader is None:
        raise RuntimeError("DataLoader is still None. Перевірте логіку ColorizationDataModule.setup()")

    metrics = ColorizationMetrics(device)

    save_dir = Path(to_absolute_path(config.output_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    images_saved = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Входи з DataLoader (діапазон [-1, 1])
            l_norms = batch["input"].to(device)
            ab_targets = batch["target"].to(device)
            hints = batch.get("hints", None)
            if hints is not None:
                hints = hints.to(device)

            # Інференс моделі (генерує [-1, 1])
            ab_preds = model(l_norms, hints)

            # ВИРІШАЛЬНИЙ ЕТАП: Конвертація обох тензорів у фізичний RGB [0, 1]
            rgb_preds_01 = batch_lab_to_rgb_metrics(l_norms, ab_preds)
            rgb_targets_01 = batch_lab_to_rgb_metrics(l_norms, ab_targets)

            # Тепер ColorizationMetrics (з data_range=1.0) отримає математично коректні дані
            metrics.update(rgb_preds_01, rgb_targets_01)

            # Збереження результатів
            if images_saved < config.save_number:
                for j in range(l_norms.size(0)):
                    if images_saved >= config.save_number: 
                        break
                    save_result_images(rgb_targets_01[j], rgb_preds_01[j], rgb_targets_01[j], save_dir, f"result_{images_saved:04d}")
                    images_saved += 1

    results = metrics.compute()
    
    print("\n" + "="*40)
    print(f"[INFO] Evaluation Results | Model: {config.model.model_name}")
    print(f"Model: {config.model.model_name.upper()}")
    print("-"*40)
    for k, v in results.items():
        print(f"  {k.upper():<10} : {v:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate()