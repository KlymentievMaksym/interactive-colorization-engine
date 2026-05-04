import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class ColorizationMetrics(nn.Module):
    """
    Клас для розрахунку об'єктивних та перцептивних метрик якості колоризації.
    УВАГА: Всі вхідні тензори ПОВИННІ бути конвертовані у простір RGB 
    з діапазоном значень [0.0, 1.0] перед передачею у метод update().
    """
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        # Фіксуємо data_range=1.0 для простору RGB.
        self.metrics = MetricCollection({
            "psnr": PeakSignalNoiseRatio(data_range=1.0),
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
            "mse": MeanSquaredError(),
            # LPIPS є стандартом де-факто для генеративних задач (CVPR, ICCV).
            # normalize=True дозволяє передавати тензори [0, 1], модуль сам переведе їх у [-1, 1] для VGG.
            "lpips": LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)
        }).to(self.device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Накопичує метрики для поточного батчу.
        
        Args:
            preds (torch.Tensor): Згенеровані RGB зображення [B, 3, H, W], діапазон [0, 1].
            target (torch.Tensor): Оригінальні RGB зображення [B, 3, H, W], діапазон [0, 1].
        """
        self.metrics.update(preds, target)

    def compute(self) -> dict[str, float]:
        """Обчислює усереднені метрики для всього набору даних."""
        results = self.metrics.compute()
        self.metrics.reset()
        return {k: v.item() for k, v in results.items()}