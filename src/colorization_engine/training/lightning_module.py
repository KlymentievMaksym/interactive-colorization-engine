import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, UniversalImageQualityIndex
from torchmetrics.regression import MeanSquaredError

from colorization_engine.utils.color_space import lab_to_rgb, denormalize_l, denormalize_ab

class LitColorizer(pl.LightningModule):
    def __init__(self, model: nn.Module, criterion: nn.Module, epochs: int = 100, lr: float = 1e-4, weight_decay: float = 1e-4, amount_show: int = 4):
        super().__init__()
        self.model = model
        self.criterion = criterion

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.amount_show = amount_show

        self.save_hyperparameters(ignore=["model", "criterion"])

        metrics = MetricCollection({
            "psnr": PeakSignalNoiseRatio(data_range=2.0),
            "ssim": StructuralSimilarityIndexMeasure(data_range=2.0),
            "mse": MeanSquaredError(),
            "uiqu": UniversalImageQualityIndex()
        })

        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/_")

    def forward(self, l_norm: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(l_norm, hints)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        l_tensor, ab_target = batch["input"], batch["target"]
        hints = batch.get("hints", None)

        ab_pred = self(l_tensor, hints)
        loss, loss_dict = self.criterion(ab_pred, ab_target, l_tensor)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True, sync_dist=True)

        if batch_idx % 16 == 0:
            self._log_images(l_tensor, ab_target, ab_pred, prefix="Train")

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        l_tensor, ab_target = batch["input"], batch["target"]
        hints = batch.get("hints", None)

        ab_pred = self(l_tensor, hints)
        
        val_loss, loss_dict = self.criterion(ab_pred, ab_target, l_tensor)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, sync_dist=True)

        self.val_metrics.update(ab_pred, ab_target)
        self.log_dict(self.val_metrics, on_epoch=True, sync_dist=True)
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()}, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            self._log_images(l_tensor, ab_target, ab_pred, prefix="Val", )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        l_tensor, ab_target = batch["input"], batch["target"]
        hints = batch.get("hints", None)

        ab_pred = self(l_tensor, hints)
        
        self.test_metrics.update(ab_pred, ab_target)
        self.log_dict(self.test_metrics, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            self._log_images(l_tensor, ab_target, ab_pred, prefix="Test")

    def _log_images(self, l_tensor: torch.Tensor, ab_target: torch.Tensor, ab_pred: torch.Tensor, prefix: str = "Val"):
        if not isinstance(self.logger, TensorBoardLogger):
            return

        self.amount_show = min(self.amount_show, l_tensor.shape[0])
        rgb_targets = []
        rgb_preds = []

        for i in range(self.amount_show):
            l_denorm = denormalize_l(l_tensor[i])           # Формат: [H, W]
            ab_t_denorm = denormalize_ab(ab_target[i])      # Формат: [H, W, 2]
            ab_p_denorm = denormalize_ab(ab_pred[i])        # Формат: [H, W, 2]

            l_expanded = np.expand_dims(l_denorm, axis=-1)
            
            lab_t = np.concatenate([l_expanded, ab_t_denorm], axis=-1) # -> [H, W, 3]
            lab_p = np.concatenate([l_expanded, ab_p_denorm], axis=-1) # -> [H, W, 3]

            rgb_t_np = lab_to_rgb(lab_t)
            rgb_p_np = lab_to_rgb(lab_p)

            rgb_targets.append(torch.from_numpy(rgb_t_np).permute(2, 0, 1).float() / 255.0)
            rgb_preds.append(torch.from_numpy(rgb_p_np).permute(2, 0, 1).float() / 255.0)

        batch_target = torch.stack(rgb_targets)
        batch_pred = torch.stack(rgb_preds)

        grid = torchvision.utils.make_grid(
            torch.cat([batch_target, batch_pred], dim=0), 
            nrow=self.amount_show
        )

        self.logger.experiment.add_image(
            f"{prefix}/First_{self.amount_show}", grid, self.current_epoch
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }