import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from colorization_engine.models import load_colorization_model
from colorization_engine.utils import TrainConfig
from colorization_engine.data import ColorizationDataModule
from colorization_engine.training.lightning_module import LitColorizer
from colorization_engine.training.losses import ColorizationLoss

CS = ConfigStore.instance()
CS.store(name="train_config", node=TrainConfig)

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(config: TrainConfig):
    print("[INFO] Initializing DataModule...")
    train_paths = [to_absolute_path(p) if isinstance(p, str) else [to_absolute_path(p[0]), to_absolute_path(p[1])] for p in config.data.train]
    val_paths = [to_absolute_path(p) if isinstance(p, str) else [to_absolute_path(p[0]), to_absolute_path(p[1])] for p in config.data.val] if config.data.val else None

    datamodule = ColorizationDataModule(
        train_paths=train_paths, val_paths=val_paths,
        image_size=config.image_size, batch_size=config.training.batch_size
    )

    print(f"[INFO] Loading model {config.model.model_name}...")
    model = load_colorization_model(config.model)
    criterion = ColorizationLoss(lambda_smooth=config.training.loss_lambda_smooth, lambda_cosine=config.training.loss_lambda_cosine)

    lit_model = LitColorizer(model=model, criterion=criterion, lr=config.training.lr, weight_decay=config.training.weight_decay)

    callbacks = []
    if config.training.do_save:
        checkpoint_callback = ModelCheckpoint(
            dirpath=to_absolute_path("checkpoints"),
            filename=f"{config.model.model_name}-{{epoch:05d}}-{{val_loss:.5f}}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True
        )
        callbacks.append(checkpoint_callback)

    print("[INFO] Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator=config.device if config.device else "auto",
        devices="auto",
        callbacks=callbacks,
        log_every_n_steps=4
        # logger=True # Lightning автоматично створить TensorBoard логер!
    )

    resume_path = to_absolute_path(config.training.resume) if config.training.resume else None
    if resume_path and os.path.isfile(resume_path):
        print(f"[INFO] Resuming training from {resume_path}...")
    else:
        resume_path = None

    trainer.fit(model=lit_model, datamodule=datamodule, ckpt_path=resume_path)

if __name__ == "__main__":
    train()