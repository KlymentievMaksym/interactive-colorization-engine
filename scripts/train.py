import torch

from scripts import load_colorization_model
from scripts.utils import Parser, TrainConfig, parse_unknown_args, get_dataloader

from colorization_engine.training.losses import ColorizationLoss
from colorization_engine.training.trainer import ColorizationTrainer


def train():
    known_args, unknown_args = Parser.train_args()
    config = TrainConfig(**vars(known_args))
    model_params = parse_unknown_args(unknown_args)

    print(f"[INFO] Loading datasets {', '.join(config.data)}{' and ' + ', '.join(config.val_data) if config.val_data else ''}...")
    train_loader = get_dataloader(config=config, is_train=True, num_workers=4)
    val_loader = get_dataloader(config=config, is_train=False, num_workers=4)

    print(f"[INFO] Loading model {config.model}...")
    model = load_colorization_model(model_name=config.model, device=torch.device(config.device), weights=config.weights, **model_params)

    criterion = ColorizationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    
    trainer = ColorizationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config.device,
        save_dir="checkpoints"
    )
    
    trainer.fit(epochs=config.epochs)

if __name__ == "__main__":
    train()