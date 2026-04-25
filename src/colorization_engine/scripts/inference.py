# src/colorization_engine/scripts/inference.py
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from colorization_engine.models import load_colorization_model
from colorization_engine.utils import InferenceConfig
from colorization_engine.data.transforms import get_transforms
from colorization_engine.data.dataset import _rgb_to_lab, _rgb_to_l_norm

class ColorizationPipeline:
    """
    Чистий API для інференсу. 
    Ідеально підходить для інтеграції у веб-додатки (colorization_app).
    """
    def __init__(self, model: torch.nn.Module, device: str | torch.device, image_size: int = 256):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.image_size = image_size
        self.transform = get_transforms(image_size=self.image_size, is_train=False)

    def preprocess(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"[ERROR] Image not found: {image_path}")

        orig_h, orig_w = img.shape[:2]
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        L_channel = _rgb_to_lab(image_rgb)[:, :, 0]

        img_resized = self.transform(image=image_rgb)['image']
        tensor_l = _rgb_to_l_norm(img_resized).unsqueeze(0)

        return tensor_l, L_channel, image_rgb, (orig_h, orig_w)

    def postprocess(self, L_channel: np.ndarray, tensor_ab: torch.Tensor, orig_shape: tuple) -> np.ndarray:
        orig_h, orig_w = orig_shape

        tensor_ab = tensor_ab.squeeze(0).cpu().detach()
        ab_denorm = tensor_ab.permute(1, 2, 0).numpy() * 110.0

        ab_upscaled = cv2.resize(ab_denorm, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        lab_result = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
        lab_result[:, :, 0] = L_channel
        lab_result[:, :, 1:] = ab_upscaled

        bgr_result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)
        bgr_result = (bgr_result * 255.0).clip(0, 255).astype(np.uint8)

        return bgr_result

    @torch.no_grad()
    def colorize(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Головний метод для додатку. 
        Приймає шлях до картинки, повертає (Кольоровий результат, Оригінал RGB).
        """
        input_tensor, L_channel, original_rgb, orig_shape = self.preprocess(image_path)

        output_ab = self.model(input_tensor.to(self.device))

        bgr_result = self.postprocess(L_channel, output_ab, orig_shape)
        return bgr_result, original_rgb


cs = ConfigStore.instance()
cs.store(name="inference_config", node=InferenceConfig)

@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def inference(config: InferenceConfig):
    image_paths = []
    
    if config.image:
        image_paths.append(Path(to_absolute_path(config.image)))

    if config.input_dir:
        dir_path = Path(to_absolute_path(config.input_dir))
        if dir_path.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
                image_paths.extend(dir_path.glob(ext))
        else:
            print(f"[WARNING] Input directory not found: {dir_path}")

    if not image_paths:
        print("[ERROR] No images found! Provide 'image=...' or 'input_dir=...'")
        return

    out_dir = Path(to_absolute_path(config.result_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = config.device if config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    print(f"[INFO] Loading model {config.model.model_name}...")
    model = load_colorization_model(config.model, device=device)
    pipeline = ColorizationPipeline(model=model, device=device, image_size=config.image_size)

    print(f"[INFO] Found {len(image_paths)} images. Starting colorization...")
    
    for img_path in tqdm(image_paths, desc="Processing Images"):
        try:
            bgr_result, original_rgb = pipeline.colorize(str(img_path))

            gray_original = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
            image_gray = cv2.cvtColor(gray_original, cv2.COLOR_GRAY2BGR)
            image_original = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

            comparison = np.hstack((image_gray, bgr_result, image_original))

            result_path = out_dir / f"result_{img_path.name}"
            cv2.imwrite(str(result_path), comparison)

        except Exception as e:
            print(f"\n[ERROR] Failed to process {img_path.name}: {e}")

    print(f"[INFO] Done! All results saved to: {out_dir}")

if __name__ == "__main__":
    inference()