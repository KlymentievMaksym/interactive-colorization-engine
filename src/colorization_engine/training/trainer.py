import os
import torch
from tqdm import tqdm

class ColorizationTrainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        device='cuda', 
        save_dir='checkpoints'
    ):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.save_dir = save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        
        # Використовуємо tqdm для красивого прогрес-бару в терміналі
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            # Наші датасети повертають словник {"input": ..., "target": ...}
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Наш loss повертає (total_loss, loss_dict)
            loss, loss_metrics = self.criterion(outputs, targets)
            
            # Backward pass та оновлення ваг
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Оновлюємо інформацію в прогрес-барі
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        for batch in pbar:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            
            outputs = self.model(inputs)
            loss, _ = self.criterion(outputs, targets)
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return running_loss / len(self.val_loader)

    def fit(self, epochs):
        print(f"[INFO] Починаємо навчання на {self.device}...")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)
            
            print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Зберігаємо найкращу модель
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
                print(f"[SAVE] Збережено найкращу модель у {save_path}")
                
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "latest_model.pth"))