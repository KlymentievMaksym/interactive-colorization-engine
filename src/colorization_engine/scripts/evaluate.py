# src/colorization_engine/scripts/evaluate.py
from pathlib import Path
from datetime import datetime

import os
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

import torch
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from colorization_engine.evaluation.metrics import ColorizationMetrics
from colorization_engine.training.lightning_module import LitColorizer
from colorization_engine.factory import build_model_pipeline
from colorization_engine.data import ColorizationDataModule
from colorization_engine.utils import EvaluateConfig
from colorization_engine.utils.color_space import kornia_lab_to_rgb

import time
import torch
import numpy as np

@torch.no_grad()
def profile_model_performance(model: torch.nn.Module, device: torch.device, image_size: int = 256, runs: int = 300):
    """Measure inference time and VRAM"""
    model.eval()
    model.to(device)

    dummy_l = torch.randn(1, 1, image_size, image_size, device=device)
    dummy_hints = torch.zeros(1, 3, image_size, image_size, device=device)

    print("[INFO] Warming up GPU for accurate timing...")
    for _ in range(10):
        _ = model(dummy_l, dummy_hints)
        
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        
    print("[INFO] Benchmarking inference time...")
    start_time = time.perf_counter()
    
    for _ in range(runs):
        _ = model(dummy_l, dummy_hints)
        
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        
    end_time = time.perf_counter()
    
    latency_ms = ((end_time - start_time) / runs) * 1000

    vram_mb = 0
    if device.type == 'cuda':
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    print(f"[INFO] Inference Time: {latency_ms:.2f} ms")
    print(f"[INFO] Peak VRAM Usage: {vram_mb:.2f} MB")

    return latency_ms, vram_mb

cs = ConfigStore.instance()
cs.store(name="evaluate_config", node=EvaluateConfig)

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def evaluate(config: EvaluateConfig):
    test_paths = [to_absolute_path(p) if isinstance(p, str) else [to_absolute_path(p[0]), to_absolute_path(p[1])] for p in config.data.test] if config.data.test else None

    if not test_paths:
        raise ValueError("[ERROR] No test data paths provided in config.data.test")

    datamodule = ColorizationDataModule(
        test_paths=test_paths,
        image_size=config.image_size, min_hint_size=config.hints.min_hint_size, max_hint_size=config.hints.max_hint_size, num_hints_val=config.hints.num_hints_val, patch_size_val=config.hints.patch_size_val,
        batch_size=config.dataloader.batch_size, num_workers=config.dataloader.num_workers, timeout=config.dataloader.timeout
    )
    
    device = torch.device(config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    print(f"[INFO] Loading model {config.model.model_name}...")
    model = build_model_pipeline(
        model_name=config.model.model_name, 
        weights_path=config.model.weights, 
        model_params=config.model.model_params, 
        device=device
    )
    model.eval()

    lit_model = LitColorizer(model=model)

    print("[INFO] Initializing PyTorch Lightning Trainer...")
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M")
    experiment_name = f"test_{config.model.model_name}_{current_time}"

    logger = TensorBoardLogger(
        save_dir="logs/",
        name="eval",
        version=experiment_name
    )
    trainer = pl.Trainer(
        logger=logger,
        precision="bf16-mixed",
        accelerator=config.device if config.device else "auto",
        enable_model_summary=True,
    )

    trainer.test(model=lit_model, datamodule=datamodule)

    profile_model_performance(model, device, image_size=config.image_size)

    if config.save_number > 0:
        print("[INFO] Saving evaluation images...")
        save_dir = Path(to_absolute_path(config.output_dir)) / config.model.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        lit_model.eval()
        lit_model.to(device)
        
        images_saved = 0
        with torch.no_grad():
            for batch in datamodule.test_dataloader(): # type: ignore
                if images_saved >= config.save_number:
                    break
                    
                l_tensor = batch["input"].to(device)
                ab_target = batch["target"].to(device)
                hints = batch.get("hints", None)
                if hints is not None:
                    hints = hints.to(device)
                    
                ab_pred = lit_model(l_tensor, hints)
                
                rgb_pred = kornia_lab_to_rgb(l_tensor, ab_pred)
                rgb_target = kornia_lab_to_rgb(l_tensor, ab_target)
                
                gray_batch = (((l_tensor + 1.0) / 2.0).repeat(1, 3, 1, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                rgb_pred_uint8 = (rgb_pred * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                rgb_target_uint8 = (rgb_target * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                
                for i in range(l_tensor.size(0)):
                    if images_saved >= config.save_number:
                        break
                        
                    gray = cv2.cvtColor(gray_batch[i].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    pred_bgr = cv2.cvtColor(rgb_pred_uint8[i].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    target_bgr = cv2.cvtColor(rgb_target_uint8[i].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    
                    comparison = np.concatenate((gray, pred_bgr, target_bgr), axis=1)
                    file_path = os.path.join(save_dir, f"result_{images_saved:04d}.jpg")
                    cv2.imwrite(file_path, comparison)
                    
                    images_saved += 1
                    
        print(f"[INFO] Saved {images_saved} images to {save_dir}")


if __name__ == "__main__":
    evaluate()