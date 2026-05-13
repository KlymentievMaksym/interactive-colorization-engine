
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity, FrechetInceptionDistance, KernelInceptionDistance
from torchmetrics.regression import MeanSquaredError

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from colorization_engine.utils.color_space import kornia_lab_to_rgb

seed_everything(42, workers=True)

class LitColorizer(pl.LightningModule):
    def __init__(self, model: nn.Module, criterion: nn.Module | None = None, epochs: int = 100, lr: float = 1e-4, weight_decay: float = 1e-4, amount_show: int = 4):
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
            "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True),
            "mse": MeanSquaredError(),
        })

        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.test_gen_metrics = MetricCollection({
            "fid": FrechetInceptionDistance(),
            "kid": KernelInceptionDistance()
        }, prefix="test/")
        self.strict_loading = False #

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        keys_to_remove = [
            k for k in checkpoint["state_dict"].keys() 
            if any(x in k for x in ["lpips", "fid", "kid", "inception"])
        ]

        for k in keys_to_remove:
            del checkpoint["state_dict"][k]

    def forward(self, l_norm: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(l_norm, hints)

    @staticmethod
    def _get_error_heatmap(pred_ab, true_ab):
        # Heatmap: Black -> Red -> Yellow -> White
        delta_ab = torch.sqrt((pred_ab - true_ab).pow(2).sum(dim=1, keepdim=True))
        error_norm = (delta_ab * 1.5).clamp(0, 1)
        r = (error_norm * 3.0).clamp(0, 1)
        g = ((error_norm - 0.333) * 3.0).clamp(0, 1)
        b = ((error_norm - 0.666) * 3.0).clamp(0, 1)
        error = torch.cat([r, g, b], dim=1)
        return error

    def on_train_start(self):
        if self.criterion is None:
            raise ValueError("Criterion is None")

        new_lr = self.lr 

        for optimizer in self.trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        if self.trainer.lr_scheduler_configs:
            for config in self.trainer.lr_scheduler_configs:
                scheduler = config.scheduler
                if hasattr(scheduler, 'base_lrs'):
                    scheduler.base_lrs = [new_lr for _ in scheduler.base_lrs]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        l_tensor, ab_target = batch["input"], batch["target"]
        hints = batch.get("hints", None)
        hint_mask = hints[:, 2:3, :, :] if hints is not None else None

        ab_pred = self(l_tensor, hints)
        loss, loss_dict = self.criterion(ab_pred, ab_target, l_tensor, hint_mask) # type: ignore

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True, sync_dist=True)

        if self.global_step % 500 == 0:
            train_example_l = l_tensor[:self.amount_show].detach()
            train_example_pred = ab_pred[:self.amount_show].detach()
            train_example_target = ab_target[:self.amount_show].detach()
            train_example_hints = hints[:self.amount_show].detach() if hints is not None else None
            self._log_train_images(train_example_l, train_example_pred, train_example_target, train_example_hints)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def _log_train_images(self, ls, preds, targets, hints: torch.Tensor | None = None):
        if not isinstance(self.logger, TensorBoardLogger):
            return

        l_channel = ls.to(self.device)
        pred_ab = preds.to(self.device)
        target_ab = targets.to(self.device)

        error = self._get_error_heatmap(pred_ab, target_ab)

        gray = ((l_channel + 1.0) / 2.0).repeat(1, 3, 1, 1)
        pred_rgb = kornia_lab_to_rgb(l_channel, pred_ab)
        target_rgb = kornia_lab_to_rgb(l_channel, target_ab)

        B, _, H, W = gray.shape

        images = [gray]
        name = "Input-"
        if hints is not None:
            hints = hints.to(self.device)
            mask_visual = hints[:, 2:3].repeat(1, 3, 1, 1)
            # hint_visual = kornia_lab_to_rgb(l_channel, hints[:, :2])

            mask = hints[:, 2:3]
            true_ab = hints[:, :2] / mask.clamp(min=1e-8)
            true_rgb = kornia_lab_to_rgb(l_channel, true_ab)
            visible_mask = (mask > 0.02).float()
            hint_visual = gray * (1 - visible_mask) + true_rgb * visible_mask

            images.extend([mask_visual, hint_visual])
            name += "Mask-Hints"
        images.extend([pred_rgb, target_rgb, error])
        name += "-Pred-True-Error"

        images_train = torch.stack(images, dim=1).view(-1, 3, H, W)
        grid_train = torchvision.utils.make_grid(images_train, nrow=len(images), padding=4, pad_value=1.0)

        self.logger.experiment.add_image(f"Train/{name}", grid_train, self.global_step)

    def on_validation_start(self):
        if self.criterion is None:
            raise ValueError("Criterion is None")
        if self.trainer.val_dataloaders is None:
            raise ValueError("Val dataloader is None")

        self.example_l = None
        self.example_ab = None
        self.example_hints = None

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        l_tensor, ab_target = batch["input"], batch["target"]
        B, _, H, W = ab_target.shape
        device = l_tensor.device

        # Auto
        empty_hints = torch.zeros((B, 3, H, W), device=device)
        ab_pred_auto = self(l_tensor, empty_hints)

        val_loss_auto, loss_dict_auto = self.criterion(ab_pred_auto, ab_target, l_tensor, hint_mask=empty_hints[:, 2:3]) # type: ignore
        self.log("val_loss_auto", val_loss_auto, on_step=True, on_epoch=True, sync_dist=True)

        rgb_pred_auto = kornia_lab_to_rgb(l_tensor, ab_pred_auto)
        rgb_target = kornia_lab_to_rgb(l_tensor, ab_target)

        self.val_metrics.update(rgb_pred_auto, rgb_target)
        self.log_dict(self.val_metrics, on_epoch=True, sync_dist=True)
        self.log_dict({f"val_auto/{k}": v for k, v in loss_dict_auto.items()}, on_epoch=True, sync_dist=True)

        # Interactive
        val_hints = batch.get("hints", None)
        
        if val_hints is not None:
            ab_pred_hinted = self(l_tensor, val_hints)

            val_loss_hinted, loss_dict_hinted = self.criterion(ab_pred_hinted, ab_target, l_tensor, hint_mask=val_hints[:, 2:3]) # type: ignore
            
            self.log("val_loss_hinted", val_loss_hinted, on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict({f"val_hinted/{k}": v for k, v in loss_dict_hinted.items()}, on_epoch=True, sync_dist=True)

            if batch_idx == 0:
                self.example_l = l_tensor[:self.amount_show].detach()
                self.example_ab = ab_target[:self.amount_show].detach()
                self.example_hints = val_hints[:self.amount_show].detach()

    def on_validation_epoch_end(self):
        if not isinstance(self.logger, TensorBoardLogger):
            return
        if self.example_l is None:
            return

        l_channel = self.example_l.to(self.device)
        true_ab = self.example_ab.to(self.device) # type: ignore
        hints = self.example_hints.to(self.device) # type: ignore

        gray = ((l_channel + 1.0) / 2.0).repeat(1, 3, 1, 1)
        true_rgb = kornia_lab_to_rgb(l_channel, true_ab)

        B, _, H, W = gray.shape

        pred_ab = self(l_channel)
        pred_ab_hints = self(l_channel, hints)

        pred_rgb = kornia_lab_to_rgb(l_channel, pred_ab)
        pred_rgb_hints = kornia_lab_to_rgb(l_channel, pred_ab_hints)

        error = self._get_error_heatmap(pred_ab, true_ab)
        error_hints = self._get_error_heatmap(pred_ab_hints, true_ab)

        # l_black = torch.full_like(l_channel, 0.0)
        hint_visual = kornia_lab_to_rgb(l_channel, hints[:, :2])

        images_auto = torch.stack([gray, pred_rgb, true_rgb, error], dim=1).view(-1, 3, H, W)
        grid_auto = torchvision.utils.make_grid(images_auto, nrow=4, padding=4, pad_value=1.0)
        self.logger.experiment.add_image("Val/Input-Pred-True-Error", grid_auto, self.current_epoch)

        images_hinted = torch.stack([hint_visual, gray, pred_rgb_hints, true_rgb, error_hints], dim=1).view(-1, 3, H, W)
        grid_hinted = torchvision.utils.make_grid(images_hinted, nrow=5, padding=4, pad_value=1.0)
        self.logger.experiment.add_image("Val/Hint-Input-HintPred-True-Error", grid_hinted, self.current_epoch)

    @staticmethod
    def _colorfulness_index(rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Colorfulness Index RGB [B, 3, H, W] [0, 1]"""
        r, g, b = rgb_tensor[:, 0], rgb_tensor[:, 1], rgb_tensor[:, 2]

        rg = torch.abs(r - g)
        yb = torch.abs(0.5 * (r + g) - b)

        std_rg = torch.std(rg, dim=[1, 2])
        mean_rg = torch.mean(rg, dim=[1, 2])
        std_yb = torch.std(yb, dim=[1, 2])
        mean_yb = torch.mean(yb, dim=[1, 2])

        colorfulness = torch.sqrt(std_rg**2 + std_yb**2) + 0.3 * torch.sqrt(mean_rg**2 + mean_yb**2)
        return colorfulness.mean()

    def on_test_start(self):
        test_loaders = self.trainer.test_dataloaders
        
        if test_loaders is not None:
            dl = test_loaders[0] if isinstance(test_loaders, list) else test_loaders
            test_size = len(dl.dataset)

            standard_subset_size = self.test_gen_metrics["kid"].subset_size
            safe_subset_size = min(standard_subset_size, test_size) # type: ignore

            self.test_gen_metrics["kid"].subset_size = safe_subset_size

            print(f"[INFO] Auto-adjusted KID subset_size to: {safe_subset_size} (Dataset size: {test_size})")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        l_tensor, ab_target = batch["input"], batch["target"]
        hints = batch.get("hints", None)

        ab_pred = self(l_tensor, hints)
        
        rgb_pred = kornia_lab_to_rgb(l_tensor, ab_pred)
        rgb_target = kornia_lab_to_rgb(l_tensor, ab_target)

        self.test_metrics.update(rgb_pred, rgb_target)

        rgb_pred_uint8 = (rgb_pred * 255).clamp(0, 255).to(torch.uint8)
        rgb_target_uint8 = (rgb_target * 255).clamp(0, 255).to(torch.uint8)

        self.test_gen_metrics.update(rgb_target_uint8, real=True)
        self.test_gen_metrics.update(rgb_pred_uint8, real=False)

        self.log_dict(self.test_metrics, on_epoch=True, sync_dist=True)

        mse_ab = torch.nn.functional.mse_loss(ab_pred, ab_target)
        self.log("test/mse_lab", mse_ab, sync_dist=True)
        self.log("test/colorfulness", self._colorfulness_index(rgb_pred), sync_dist=True)

        if hints is not None:
            mask = hints[:, 2:3] # [B, 1, H, W]
            hint_error = torch.nn.functional.mse_loss(ab_pred * mask, ab_target * mask, reduction='sum')
            hint_mse = hint_error / (mask.sum() * 2 + 1e-8) 
            self.log("test/hint_mse", hint_mse, sync_dist=True)

    def on_test_epoch_end(self):
        gen_results = self.test_gen_metrics.compute()

        flat_results = {}
        for key, value in gen_results.items():
            if isinstance(value, tuple):
                flat_results[f"{key}_mean"] = value[0]
                flat_results[f"{key}_std"] = value[1]
            else:
                flat_results[key] = value

        self.log_dict(flat_results)
        self.test_gen_metrics.reset()

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