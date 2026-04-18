import os
import argparse
import torch
from torch.utils.data import DataLoader

# Імпортуємо наші модулі
from colorization_engine.data_loaders.dataset import PairedDataset, SingleTargetFolderDataset
from colorization_engine.data_loaders.transforms import get_transforms
from colorization_engine.models.my_colorization import Colorization
from colorization_engine.training.losses import ColorizationLoss
from colorization_engine.training.trainer import ColorizationTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Тренування моделі колоризації на базі Mamba-SSM")
    parser.add_argument("--data_dir", type=str, required=True, help="Шлях до папки з даними")
    parser.add_argument("--weights", type=str, default="checkpoints/latest_model.pth", help="Шлях до ваг моделі")
    parser.add_argument("--epochs", type=int, default=50, help="Кількість епох")
    parser.add_argument("--batch_size", type=int, default=8, help="Розмір батчу")
    parser.add_argument("--d_model", type=int, default=256, help="Розмір прихованого стану")
    parser.add_argument("--image_size", type=int, default=256, help="Розмір зображення")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("[INFO] Ініціалізація датасетів...")
    transform = get_transforms(image_size=args.image_size, is_train=True)
    
    train_dataset = SingleTargetFolderDataset(args.data_dir, transform)
    val_dataset = PairedDataset(
        dir_inputs="../data/NCD/NCD Dataset/Gray", 
        dir_targets="../data/NCD/NCD Dataset/Color", 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print("[INFO] Ініціалізація моделі...")
    model = Colorization(d_model=args.d_model, layers=6, blocks=2)
    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=args.device)
        # Якщо ми зберегли словник (best_model.pth), дістаємо з нього. Якщо просто ваги - беремо напряму.
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print(f"[INFO] Ваги {args.weights} успішно завантажено!")
    
    criterion = ColorizationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    trainer = ColorizationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        save_dir="checkpoints"
    )
    
    trainer.fit(epochs=args.epochs)

if __name__ == "__main__":
    main()