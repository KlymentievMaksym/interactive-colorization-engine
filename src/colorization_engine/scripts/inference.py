import os
import torch
import cv2
import numpy as np

from colorization_engine.scripts import load_colorization_model
from colorization_engine.scripts.utils import Parser, InferenceConfig, parse_unknown_args
from colorization_engine.data_loaders.transforms import get_transforms
from colorization_engine.data_loaders.dataset import __rgb_to_lab, _rgb_to_l_norm


def preprocess_image_lab(image_path: str, image_size: int):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[Error] Image not found: {image_path}")
    
    orig_h, orig_w = img.shape[:2]
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    L_channel = __rgb_to_lab(image_rgb)[:, :, 0]
    
    img_resized = get_transforms(image_size=image_size, is_train=False)(image=image_rgb, target=None)['image']
    # img_resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    tensor_l = _rgb_to_l_norm(img_resized).unsqueeze(0)

    return tensor_l, L_channel, image_rgb, (orig_h, orig_w)

def postprocess_lab_tensor(L_channel, tensor_ab, orig_shape):
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

def inference():
    known_args, unknown_args = Parser.inference_args()
    config = InferenceConfig(**vars(known_args))
    model_params = parse_unknown_args(unknown_args)

    device = torch.device(config.device)
    print(f"[INFO] Loading model {config.model}...")
    model = load_colorization_model(model_name=config.model, device=device, weights=config.weights, **model_params)
    model.eval()

    if config.result is None:
        file_name = os.path.basename(config.image)
        config.result = f"result_{file_name}"

    print(f"[INFO] Preprocessing image {config.image}...")
    input_tensor, L_channel, original_rgb, orig_shape = preprocess_image_lab(config.image, image_size=config.image_size)

    print(f"[INFO] Predicting image a and b...")
    with torch.no_grad():
        output_ab = model(input_tensor.to(device))

    print(f"[INFO] Postrocessing image...")
    image_result = postprocess_lab_tensor(L_channel, output_ab, orig_shape)

    gray_original = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    image_gray = cv2.cvtColor(gray_original, cv2.COLOR_GRAY2BGR)
    image_original = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

    comparison = np.hstack((image_gray, image_result, image_original))
    cv2.imwrite(config.result, comparison)
    print(f"[INFO] Done! Result saved to: {config.result}")

if __name__ == "__main__":
    inference()