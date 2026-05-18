# src/colorization_engine/scripts/inference.py
import os
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import torch
torch.set_num_threads(2)

import numpy as np
from pathlib import Path
from tqdm import tqdm
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from colorization_engine.models.util_models import BaseColorizer
from colorization_engine.factory import build_model_pipeline
from colorization_engine.utils import InferenceConfig
from colorization_engine.utils.color_space import rgb_to_lab, normalize_l, denormalize_ab

class ColorizationPipeline:
    """
    Inference API
    """
    def __init__(self, model: BaseColorizer, device: str | torch.device, image_size: int = 256):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.image_size = image_size

    def preprocess(self, image: np.ndarray, hints: np.ndarray | None = None, color_intensity: float = 1.0):
        orig_h, orig_w = image.shape[:2]

        image_lab = rgb_to_lab(image)
        L_channel = image_lab[:, :, 0]
        l_channel_resized = cv2.resize(L_channel, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor_l = normalize_l(l_channel_resized).unsqueeze(0)

        tensor_hints = None

        if hints is not None:
            # 1. Знаходимо маску на ОРИГІНАЛІ (щоб не розмити колір при ресайзі)
            hints_alpha = hints[:, :, 3] > 0 # Булева маска намальованого
            
            # Якщо нічого не намальовано
            if not np.any(hints_alpha):
                return tensor_l, None, L_channel, (orig_h, orig_w)

            # 2. Створюємо порожній AB тензор для підказок
            ab_hint_tensor = np.zeros((self.image_size, self.image_size, 2), dtype=np.float32)
            mask_tensor = np.zeros((self.image_size, self.image_size, 1), dtype=np.float32)

            # 3. Витягуємо кольори та їх координати
            # Знаходимо контури крапок, щоб працювати з кожною окремо
            uint8_alpha = np.where(hints[:, :, 3] > 0, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(uint8_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # ФІКС 2: Множимо на 255.0! Інакше кольори [0, 1] перетворюються на чорний бруд.
                color_rgb = (hints[cY, cX, :3] * 255.0).clip(0, 255).astype(np.uint8) 
                
                color_rgb_1x1 = color_rgb.reshape(1, 1, 3)
                color_lab = rgb_to_lab(color_rgb_1x1)
                color_ab_norm = (color_lab[0, 0, 1:] / 110.0) * color_intensity 
                color_ab_norm = np.clip(color_ab_norm, -1.0, 1.0)

                scale_x = self.image_size / orig_w
                scale_y = self.image_size / orig_h
                tensor_cX = int(cX * scale_x)
                tensor_cY = int(cY * scale_y)

                area = cv2.contourArea(cnt)
                scaled_area = area * (scale_x * scale_y)
                dynamic_radius = int(np.sqrt(scaled_area / np.pi))
                
                radius = np.clip(dynamic_radius, 2, 12)
                y1, y2 = max(0, tensor_cY - radius), min(self.image_size, tensor_cY + radius + 1)
                x1, x2 = max(0, tensor_cX - radius), min(self.image_size, tensor_cX + radius + 1)

                for y in range(y1, y2):
                    for x in range(x1, x2):
                        dist_sq = (x - tensor_cX)**2 + (y - tensor_cY)**2
                        if dist_sq <= radius**2:
                            sigma = radius / 4.0
                            weight = np.exp(-dist_sq / (2 * sigma**2))
                            
                            # ФІКС 3: Оновлюємо колір тільки якщо цей гауссовий піксель сильніший за попередній 
                            # (щоб точки, які близько одна до одної, не змішувалися в кашу)
                            if weight > mask_tensor[y, x, 0]:
                                mask_tensor[y, x, 0] = weight
                                ab_hint_tensor[y, x] = color_ab_norm

            # 4. Об'єднуємо і створюємо тензор
            # Множимо AB на маску, як у лодері (hint_ab = ab_tensor * mask)
            hints_combined = np.concatenate([ab_hint_tensor * mask_tensor, mask_tensor], axis=-1)
            tensor_hints = torch.from_numpy(hints_combined).permute(2, 0, 1).unsqueeze(0).float()

        return tensor_l, tensor_hints, L_channel, (orig_h, orig_w)

    def postprocess(self, L_channel: np.ndarray, tensor_ab: torch.Tensor, orig_shape: tuple) -> np.ndarray:
        orig_h, orig_w = orig_shape  # (H, W)

        ab_denorm = denormalize_ab(tensor_ab.squeeze(0))  # [IS, IS, 2]

        ab_upscaled = cv2.resize(ab_denorm, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)  # [H, W, 2]

        # [H, W, 2] + [H, W, 1] -> [H, W, 3]
        lab_result = np.empty((orig_h, orig_w, 3), dtype=np.float32)
        lab_result[:, :, 0] = L_channel
        lab_result[:, :, 1:] = ab_upscaled
        rgb_result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2RGB)
        rgb_result = (rgb_result * 255.0).clip(0, 255).astype(np.uint8)

        # [H, W, 3]
        return rgb_result

    @torch.no_grad()
    def colorize(self, image: np.ndarray, hints: np.ndarray | None = None, color_intensity: float = 1.0, num_samples: int = 5) -> list[np.ndarray]:
        """
        Головний метод для додатку. 
        Приймає картинку, повертає (Кольорові результати).
        """
        input_tensor, tensor_hints, L_channel, orig_shape = self.preprocess(image, hints, color_intensity)

        input_tensor = input_tensor.to(self.device)
        tensor_hints = tensor_hints.to(self.device) if tensor_hints is not None else None

        output_abs = self.model.sample(input_tensor, tensor_hints, num_samples=num_samples, color_intensity=color_intensity)

        results = []
        for output_ab in output_abs:
            rgb_result = self.postprocess(L_channel, output_ab.float(), orig_shape)  # [H, W, 3]
            results.append(rgb_result)

        return results


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

    device_name = config.device if config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    print(f"[INFO] Loading model {config.model.model_name}...")
    model_config = config.model
    model = build_model_pipeline(model_name=model_config.model_name, weights_path=model_config.weights, model_params=model_config.model_params, device=device)
    pipeline = ColorizationPipeline(model=model, device=device, image_size=config.image_size)

    out_dir = Path(to_absolute_path(config.result_dir)) / model_config.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(image_paths)} images. Starting colorization with image size {config.image_size}...")

    for img_path in tqdm(image_paths, desc="Processing Images"):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"[ERROR] Image not found: {img_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            results = pipeline.colorize(img_rgb)

            gray_original = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            image_gray = cv2.cvtColor(gray_original, cv2.COLOR_GRAY2BGR)
            image_original = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            comparison = np.hstack((image_gray, image_original))

            img_errs = []
            for rgb_result in results:
                bgr_result = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2BGR)

                t_res = torch.from_numpy(rgb_result).float() / 255.0
                t_orig = torch.from_numpy(img_rgb).float() / 255.0
                delta_ab = torch.sqrt((t_res - t_orig).pow(2).sum(dim=-1, keepdim=True))

                error_norm = (delta_ab * 1.5).clamp(0, 1)
                r = (error_norm * 3.0).clamp(0, 1)
                g = ((error_norm - 0.333) * 3.0).clamp(0, 1)
                b = ((error_norm - 0.666) * 3.0).clamp(0, 1)
                
                error = torch.cat([r, g, b], dim=-1).numpy()
                
                error_uint8 = (error * 255.0).astype(np.uint8)
                image_error = cv2.cvtColor(error_uint8, cv2.COLOR_RGB2BGR)
                img_errs.append(np.hstack((bgr_result, image_error)))

            comparison = np.vstack((comparison, *img_errs))

            result_path = out_dir / f"result_{img_path.name}"
            cv2.imwrite(str(result_path), comparison)

        except Exception as e:
            print(f"\n[ERROR] Failed to process {img_path.name}: {e}")

    print(f"[INFO] Done! All results saved to: {out_dir}")

if __name__ == "__main__":
    inference()