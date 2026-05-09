import numpy as np

import kornia
# from skimage.color import lab2rgb

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from matplotlib.figure import Figure

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, UniversalImageQualityIndex , LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

# from colorization_engine.utils.color_space import lab_batch_to_rgb


seed_everything(42, workers=True)

def kornia_lab_to_rgb(l_channel: torch.Tensor, ab_channels: torch.Tensor) -> torch.Tensor:
    """
    Конвертує нормалізовані L [-1, 1] та AB [-1, 1] тензори в RGB [0, 1] 
    повністю на GPU зі збереженням графів.
    """
    l_unnorm = (l_channel + 1.0) * 50.0
    ab_unnorm = ab_channels * 110.0
    
    lab = torch.cat([l_unnorm, ab_unnorm], dim=1)
    # kornia повертає RGB тензор у діапазоні [0.0, 1.0]
    return kornia.color.lab_to_rgb(lab)

# def make_panel(gray: torch.Tensor, pred: torch.Tensor, true: torch.Tensor) -> Figure:
#     error = (pred - true).abs().mean(dim=0)

#     # 1. Збільшуємо розмір полотна та DPI (якість відмальовки)
#     fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)

#     images = [gray, pred, true, error]
#     titles = ["Input", "Prediction", "Ground Truth", "Error"]

#     for ax, img, title in zip(axes, images, titles):
#         # matplotlib краще працює з numpy масивами
#         if img.dim() == 2:
#             ax.imshow(img.cpu().numpy(), cmap="gray")
#         elif img.shape[0] == 1:
#             ax.imshow(img.squeeze().cpu().numpy(), cmap="gray")
#         else:
#             # Не забуваємо clamp(0, 1), щоб matplotlib не сварився на float
#             ax.imshow(img.permute(1, 2, 0).cpu().clamp(0, 1).numpy())

#         # 2. Робимо заголовки читабельними
#         ax.set_title(title, fontsize=16, pad=12)
#         ax.axis("off")

#     # 3. Агресивно вбиваємо всі білі відступи навколо картинок
#     plt.tight_layout()
#     fig.subplots_adjust(wspace=0.05, hspace=0, left=0.02, right=0.98, bottom=0.02, top=0.88)
    
#     return fig


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
            "psnr": PeakSignalNoiseRatio(data_range=1.0),
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
            "mse": MeanSquaredError(),
            "uiqu": UniversalImageQualityIndex(),
            "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        })

        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.strict_loading = False 

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Strip the LPIPS metric weights from the checkpoint to prevent bloat."""
        keys_to_remove = [k for k in checkpoint["state_dict"].keys() if "lpips_metric" in k]
        for k in keys_to_remove:
            del checkpoint["state_dict"][k]

    def forward(self, l_norm: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(l_norm, hints)

    def on_validation_start(self):
        if self.trainer.val_dataloaders is None:
            raise ValueError("Val dataloader is None")

        batch = next(iter(self.trainer.val_dataloaders))
        self.example_l = batch["input"][:self.amount_show]
        self.example_ab = batch["target"][:self.amount_show]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        l_tensor, ab_target = batch["input"], batch["target"]
        hints = batch.get("hints", None)
        hint_mask = hints[:, 2:3, :, :] if hints is not None else None

        ab_pred = self(l_tensor, hints)
        loss, loss_dict = self.criterion(ab_pred, ab_target, l_tensor, hint_mask)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        l_tensor, ab_target = batch["input"], batch["target"]
        hints = batch.get("hints", None)
        hint_mask = hints[:, 2:3, :, :] if hints is not None else None

        ab_pred = self(l_tensor, hints)
        
        val_loss, loss_dict = self.criterion(ab_pred, ab_target, l_tensor, hint_mask)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, sync_dist=True)

        rgb_pred = kornia_lab_to_rgb(l_tensor, ab_pred)
        rgb_target = kornia_lab_to_rgb(l_tensor, ab_target)

        self.val_metrics.update(rgb_pred, rgb_target)
        self.log_dict(self.val_metrics, on_epoch=True, sync_dist=True)
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()}, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        if not isinstance(self.logger, TensorBoardLogger):
            return
            
        l_channel = self.example_l.to(self.device)
        true_ab = self.example_ab.to(self.device)

        pred_ab = self(l_channel)

        pred_rgb = kornia_lab_to_rgb(l_channel, pred_ab)
        true_rgb = kornia_lab_to_rgb(l_channel, true_ab)
        
        gray = ((l_channel + 1.0) / 2.0).repeat(1, 3, 1, 1)
        
        # error = (pred_rgb - true_rgb).abs().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        delta_ab = torch.sqrt((pred_ab - true_ab).pow(2).sum(dim=1, keepdim=True))

        # 2. Нормалізуємо помилку від 0 до 1. 
        # Множимо на 1.5, щоб підсилити контраст слабких помилок (це як "експозиція")
        error_norm = (delta_ab * 1.5).clamp(0, 1)

        # 3. Генеруємо теплову карту "Fire" (Чорний -> Червоний -> Жовтий -> Білий) повністю на GPU
        # Червоний канал: швидко зростає від 0 до 0.33
        r = (error_norm * 3.0).clamp(0, 1)
        # Зелений канал: починає рости після 0.33 і до 0.66 (дає жовтий колір у сумі з червоним)
        g = ((error_norm - 0.333) * 3.0).clamp(0, 1)
        # Синій канал: починає рости після 0.66 (дає білий колір у сумі з червоним і зеленим)
        b = ((error_norm - 0.666) * 3.0).clamp(0, 1)

        # Збираємо кольорову карту [B, 3, H, W]
        error = torch.cat([r, g, b], dim=1)

        B, _, H, W = gray.shape
        images = torch.stack([gray, pred_rgb, true_rgb, error], dim=1).view(-1, 3, H, W)

        grid = torchvision.utils.make_grid(images, nrow=4, padding=4, pad_value=1.0)

        self.logger.experiment.add_image("Val/[Input | Pred | True | Error]", grid, self.current_epoch)
        # for i in range(min(self.amount_show, l_channel.size(0))):
        #     # Збираємо всі 4 картинки в один батч [4, 3, H, W]
        #     images = torch.stack([
        #         gray[i],          # Input
        #         pred_rgb[i],      # Prediction
        #         true_rgb[i],      # Ground Truth
        #         error[i]          # Error Map
        #     ])

        #     # nrow=4 (в один ряд), padding=4 (відступ між картинками), pad_value=1.0 (білий колір рамки)
        #     grid = torchvision.utils.make_grid(images, nrow=4, padding=4, pad_value=1.0)
        #     tag_name = f"Val/sample_{i}_[Input | Pred | True | Error]"
        #     self.logger.experiment.add_image(tag_name, grid, self.current_epoch)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        l_tensor, ab_target = batch["input"], batch["target"]
        hints = batch.get("hints", None)

        ab_pred = self(l_tensor, hints)
        
        rgb_pred = kornia_lab_to_rgb(l_tensor, ab_pred)
        rgb_target = kornia_lab_to_rgb(l_tensor, ab_target)
        
        self.test_metrics.update(rgb_pred, rgb_target)
        self.log_dict(self.test_metrics, on_epoch=True, sync_dist=True)

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