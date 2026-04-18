import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image

from colorization_engine.models.my_colorization import Colorization

def parse_args():
    parser = argparse.ArgumentParser(description="Інференс моделі колоризації")
    parser.add_argument("--image", type=str, required=True, help="Шлях до вхідної картинки")
    parser.add_argument("--weights", type=str, default="checkpoints/latest_model.pth", help="Шлях до ваг моделі")
    parser.add_argument("--out", type=str, default="result.jpg", help="Куди зберегти результат")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d_model", type=int, default=256, help="Розмір прихованого стану")
    return parser.parse_args()

def preprocess_image(image_path, size=256):
    """Завантажує картинку, робить її Ч/Б і готує для PyTorch"""
    # Завантажуємо через OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не знайдено картинку: {image_path}")
    
    # Робимо чорно-білою (імітуємо вхідні дані)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Повертаємо 3 канали, бо твій Енкодер очікує nn.Conv2d(3, 64, ...)
    # Просто дублюємо Ч/Б канал тричі
    gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # Ресайз до 256x256 (як під час тренування)
    gray_resized = cv2.resize(gray_3c, (size, size))
    
    # Нормалізуємо до [-1, 1], як у твоєму Normalize
    tensor = torch.from_numpy(gray_resized).float().permute(2, 0, 1) # HWC -> CHW
    tensor = (tensor / 127.5) - 1.0 
    
    return tensor.unsqueeze(0), img # Повертаємо тензор і оригінал для порівняння

def postprocess_tensor(tensor, size):
    """Повертає тензор [-1, 1] назад у формат картинки [0, 255]"""
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1.0) / 2.0 # [-1, 1] -> [0, 1]
    tensor = tensor.clamp(0, 1) * 255.0 # [0, 1] -> [0, 255]
    
    img_np = tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    # Повертаємо оригінальний розмір
    img_np = cv2.resize(img_np, (size[1], size[0])) 
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def main():
    args = parse_args()
    print(f"[INFO] Запуск на пристрої: {args.device}")

    # 1. Завантажуємо модель
    print("[INFO] Ініціалізація Mamba-моделі...")
    model = Colorization(d_model=args.d_model, layers=6, blocks=2).to(args.device)
    
    # 2. Завантажуємо ваги
    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=args.device)
        # Якщо ми зберегли словник (best_model.pth), дістаємо з нього. Якщо просто ваги - беремо напряму.
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print(f"[INFO] Ваги {args.weights} успішно завантажено!")
    else:
        print(f"[WARNING] Файл ваг {args.weights} не знайдено! Модель видасть випадковий шум.")

    model.eval()

    # 3. Обробка картинки
    print(f"[INFO] Обробка картинки: {args.image}")
    input_tensor, original_img = preprocess_image(args.image)
    input_tensor = input_tensor.to(args.device)

    # 4. Магія! (Forward pass)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 5. Зберігаємо результат
    h, w = original_img.shape[:2]
    result_img = postprocess_tensor(output_tensor, size=(h, w))
    
    # Склеюємо оригінал (ч/б) і результат поруч для наочності
    gray_original = cv2.cvtColor(cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    comparison = np.hstack((gray_original, result_img))
    
    cv2.imwrite(args.out, comparison)
    print(f"[SUCCESS] Готово! Результат збережено у: {args.out}")

if __name__ == "__main__":
    main()