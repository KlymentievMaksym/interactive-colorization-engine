import os
import torch
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from colorization_engine.models import load_colorization_model
from colorization_engine.utils import TrainConfig
from colorization_engine.data import get_dataloader
from colorization_engine.training.losses import ColorizationLoss
from colorization_engine.training.trainer import ColorizationTrainer

CS = ConfigStore.instance()
CS.store(name="train_config", node=TrainConfig)

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(config: TrainConfig):
    device_name = config.device if config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    print(f"[INFO] Loading model {config.model.model_name}...")
    model = load_colorization_model(config.model)

    print(f"[INFO] Loading datasets...")
    train_paths = [to_absolute_path(p) for p in config.data.train]
    val_paths = [to_absolute_path(p) for p in config.data.val] if config.data.val else None

    train_loader = get_dataloader(data_paths=train_paths, image_size=config.image_size, is_train=True, batch_size=config.training.batch_size)
    val_loader = get_dataloader(data_paths=val_paths, image_size=config.image_size, is_train=False, batch_size=config.training.batch_size) if val_paths else None

    print(f"[INFO] Initializing Optimizer & Loss (lr={config.training.lr})")
    criterion = ColorizationLoss(lambda_smooth=config.training.loss_lambda_smooth, lambda_cosine=config.training.loss_lambda_cosine)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)

    start_epoch = 1
    best_val_loss = float('inf')

    if config.training.resume:
        resume_path = to_absolute_path(config.training.resume)
        if os.path.isfile(resume_path):
            print(f"[INFO] Resuming training from {resume_path}...")
            checkpoint = torch.load(resume_path, map_location=device)

            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("[INFO] Model weights loaded successfully!")

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('val_loss', best_val_loss)

    trainer = ColorizationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        do_save=config.training.do_save,
        save_name=config.model.model_name,
        save_dir="checkpoints",
        plot_dir="results/train"
    )

    trainer.fit(epochs=config.training.epochs, start_epoch=start_epoch, best_val_loss=best_val_loss)

if __name__ == "__main__":
    train()