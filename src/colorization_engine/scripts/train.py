import os
from datetime import datetime

import faulthandler
faulthandler.enable()

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
torch.set_float32_matmul_precision('medium')
torch.set_num_threads(1)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from colorization_engine.factory import build_model_pipeline, build_loss
from colorization_engine.utils import TrainConfig
from colorization_engine.data import ColorizationDataModule
from colorization_engine.training.lightning_module import LitColorizer

CS = ConfigStore.instance()
CS.store(name="train_config", node=TrainConfig)

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(config: TrainConfig):
    print("[INFO] Initializing DataModule...")
    train_paths = [to_absolute_path(p) if isinstance(p, str) else [to_absolute_path(p[0]), to_absolute_path(p[1])] for p in config.data.train]
    val_paths = [to_absolute_path(p) if isinstance(p, str) else [to_absolute_path(p[0]), to_absolute_path(p[1])] for p in config.data.val] if config.data.val else None

    datamodule = ColorizationDataModule(
        train_paths=train_paths, val_paths=val_paths,
        image_size=config.image_size, hint_size=config.hint_size,
        batch_size=config.training.batch_size, num_workers=config.training.num_workers, timeout=config.training.timeout
    )

    print(f"[INFO] Loading model {config.model.model_name}...")
    device = config.device if config.device is not None else "cpu"
    model_config = config.model
    loss_config = config.loss

    model = build_model_pipeline(model_name=model_config.model_name, weights_path=model_config.weights, model_params=model_config.model_params, device=device)
    criterion = build_loss(loss_name=loss_config.loss_name, loss_params=loss_config.loss_params)
    # model = config.model
    # model = load_colorization_model(model_name=model.model_name, weights=model.weights, model_params=model.model_params, device=config.device)
    # # criterion = ColorizationLoss(lambda_l1=config.training.loss_lambda_l1, lambda_cos=config.training.loss_lambda_cos, lambda_sat=config.training.loss_lambda_sat, color_weight=config.training.loss_color_weight)
    # criterion = ColorizationLoss(device=config.device)

    lit_model = LitColorizer(model=model, criterion=criterion, lr=config.training.lr, weight_decay=config.training.weight_decay, amount_show=config.training.amount_show)

    callbacks = []
    if config.training.do_save:
        checkpoint_best = ModelCheckpoint(
            dirpath=to_absolute_path("checkpoints"),
            filename=f"{config.model.model_name}-best-{{epoch:02d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        )
        checkpoint_last = ModelCheckpoint(
            dirpath=to_absolute_path("checkpoints"),
            filename="train_last",
            save_on_train_epoch_end=True
        )
        # early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min")
        callbacks.extend([checkpoint_best, checkpoint_last])  # , early_stop

    print("[INFO] Initializing PyTorch Lightning Trainer...")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    experiment_name = f"{config.model.model_name}_ep{config.training.epochs:04d}_params{trainable_params / 1_000_000:.1f}M_{current_time}"

    logger = TensorBoardLogger(
        save_dir="logs/",
        name="colorization_engine",
        version=experiment_name
    )
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.training.epochs,
        accelerator=config.device if config.device else "auto",
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=1.0
    )

    resume_path = to_absolute_path(config.training.resume) if config.training.resume else None
    if resume_path and os.path.isfile(resume_path):
        print(f"[INFO] Resuming training from {resume_path}...")
    else:
        resume_path = None

    trainer.fit(model=lit_model, datamodule=datamodule, ckpt_path=resume_path)

if __name__ == "__main__":
    train()