import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError

class ColorizationMetrics:
    def __init__(self, device: torch.device):
        # PSNR та SSIM — це база для будь-якої наукової роботи з зображеннями
        self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device) # діапазон [-1, 1] має довжину 2.0
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.mse = MeanSquaredError().to(device)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Додає результати батчу в накопичувач.
        Тензори мають бути в [-1, 1]
        """
        self.psnr.update(preds, target)
        self.ssim.update(preds, target)
        self.mse.update(preds, target)

    def compute(self) -> dict[str, float]:
        """Обчислює фінальні значення по всьому набору даних."""
        return {
            "psnr": float(self.psnr.compute()),
            "ssim": float(self.ssim.compute()),
            "mse": float(self.mse.compute())
        }

    def reset(self):
        """Очищає стан (наприклад, після закінчення епохи)."""
        self.psnr.reset()
        self.ssim.reset()
        self.mse.reset()