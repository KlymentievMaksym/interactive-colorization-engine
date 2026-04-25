import os
import torch

from colorization_engine.models import load_colorization_model
from colorization_engine.utils import Parser, TrainConfig, parse_unknown_args, get_dataloader

from colorization_engine.training.losses import ColorizationLoss
from colorization_engine.training.trainer import ColorizationTrainer


def train():
    known_args, unknown_args = Parser.train_args()
    config = TrainConfig(**vars(known_args))
    model_params = parse_unknown_args(unknown_args)

    print(f"[INFO] Loading model {config.model}...")
    # print(config)
    model, standard_config = load_colorization_model(model_name=config.model, device=torch.device(config.device), weights_path=config.weights, config_path=config.config, **model_params)

    lr = standard_config.training.lr if hasattr(standard_config.training, 'lr') else None
    epochs = standard_config.training.epochs if hasattr(standard_config.training, 'epochs') else None
    batch_size = standard_config.training.batch_size if hasattr(standard_config.training, 'batch_size') else None

    config.lr = config.lr if config.lr is not None else lr
    config.epochs = config.epochs if config.epochs is not None else epochs
    config.batch_size = config.batch_size if config.batch_size is not None else batch_size
    config.image_size = config.image_size if config.image_size is not None else standard_config.image_size

    # print(config)
    # raise NotImplementedError

    print(f"[INFO] Loading datasets {', '.join(config.data)}{' and ' + ', '.join(config.val_data) if config.val_data else ''} with image size {config.image_size} ...")
    train_loader = get_dataloader(config=config, is_train=True, num_workers=4)
    val_loader = get_dataloader(config=config, is_train=False, num_workers=4)

    print(f"[INFO] Loading loss, while also adding lr: {config.lr}...")
    criterion = ColorizationLoss(.5, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    start_epoch = 1
    best_val_loss = float('inf')

    if config.resume and os.path.isfile(config.resume):
        print(f"[INFO] Resuming training from {config.resume}...")
        checkpoint = torch.load(config.resume, map_location=config.device)

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', start_epoch)
        best_val_loss = checkpoint.get('val_loss', best_val_loss)

    trainer = ColorizationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config.device,
        do_save=config.no_save,
        save_dir="checkpoints"
    )

    trainer.fit(epochs=config.epochs, start_epoch=start_epoch, best_val_loss=best_val_loss)

if __name__ == "__main__":
    train()