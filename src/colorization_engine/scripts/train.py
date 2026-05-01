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
    model = config.model
    model = load_colorization_model(model_name=model.model_name, weights=model.weights, model_params=model.model_params, device=config.device)
    # criterion = ColorizationLoss(lambda_l1=config.training.loss_lambda_l1, lambda_cos=config.training.loss_lambda_cos, lambda_sat=config.training.loss_lambda_sat, color_weight=config.training.loss_color_weight)
    criterion = ColorizationLoss(device=config.device)

    lit_model = LitColorizer(model=model, criterion=criterion, lr=config.training.lr, weight_decay=config.training.weight_decay, amount_show=config.training.amount_show)

    callbacks = []
    if config.training.do_save:
        checkpoint_best = ModelCheckpoint(
            dirpath=to_absolute_path("checkpoints"),
            filename=f"{config.model.model_name}-best-{{epoch:02d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            save_weights_only=True
        )
        
        checkpoint_last = ModelCheckpoint(
            dirpath=to_absolute_path("checkpoints"),
            filename="train_last",
            save_on_train_epoch_end=True
        )
        
        callbacks.extend([checkpoint_best, checkpoint_last])

    print("[INFO] Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator=config.device if config.device else "auto",
        callbacks=callbacks,
        log_every_n_steps=4,
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