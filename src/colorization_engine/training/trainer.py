import os
import torch
import matplotlib
# ВАЖЛИВО: Використовуємо 'Agg' для рендеру у файл без створення GUI вікон
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torchvision
import numpy as np
import cv2
from tqdm import tqdm

class ColorizationTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device: torch.device | str = 'cpu', do_save: bool = True, save_name: str | None = None, save_dir: str = 'checkpoints', plot_dir: str = 'results/train'):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.do_save = do_save
        self.save_name = save_name if save_name is not None else "model"
        self.save_dir = save_dir
        self.plot_dir = plot_dir

        self.train_loss_history = []
        self.val_loss_history = []

        self.fig, (self.ax_img, self.ax_loss) = plt.subplots(1, 2, figsize=(20, 8))

        self.ax_img.axis('off')
        self.ax_img.set_title("Validation: [Input L | Prediction ab | Target ab]")
        self.img_plot = None

        self.ax_loss.set_title("Training Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True)
        self.ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))

        self.train_line, = self.ax_loss.plot([], [], label="Train Loss", color="blue", marker="o", markersize=4)
        self.val_line, = self.ax_loss.plot([], [], label="Val Loss", color="orange", marker="o", markersize=4)
        self.ax_loss.legend()

        if self.do_save:
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.plot_dir, exist_ok=True)

    def _receive_images(self, batch):
        inputs = batch["input"].to(self.device)
        targets = batch["target"].to(self.device)
        return inputs, targets

    def _receive_loss(self, inputs, targets):
        outputs = self.model(inputs)
        return self.criterion(outputs, targets)

    def _update_loss(self, loss, loss_metrics, batches_pbar):
        self.running_loss += loss.item()
        batches_pbar.set_postfix({"loss": f"{loss.item():.4f}"})#, "metrics": loss_metrics})

    def _lab_to_rgb(self, input_img, target_img):
        _input_img = (input_img + 1.0) * 50.0
        _target_img = target_img * 110.0
        
        lab_tensor = torch.cat([_input_img, _target_img], dim=1)
        
        B = lab_tensor.shape[0]
        rgb_tensors = []
        
        for b in range(B):
            lab_np = lab_tensor[b].permute(1, 2, 0).cpu().numpy().astype(np.float32)
            rgb_np = cv2.cvtColor(lab_np, cv2.COLOR_Lab2RGB)
            rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
            rgb_tensors.append(rgb_tensor)
            
        return torch.stack(rgb_tensors).to(self.device)

    def train_epoch(self, epoch):
        self.model.train()
        self.running_loss = 0.0

        batches_pbar = tqdm(self.train_loader, desc=f"[Epoch] {epoch} [Train]")
        
        for batch in batches_pbar:
            inputs, targets = self._receive_images(batch)

            self.optimizer.zero_grad()

            loss, loss_metrics = self._receive_loss(inputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self._update_loss(loss, loss_metrics, batches_pbar)

        return self.running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        self.running_loss = 0.0
        
        batches_pbar = tqdm(self.val_loader, desc=f"[Epoch] {epoch} [Val]")
        
        for batch in batches_pbar:
            inputs, targets = self._receive_images(batch)

            loss, loss_metrics = self._receive_loss(inputs, targets)

            self._update_loss(loss, loss_metrics, batches_pbar)

        return self.running_loss / len(self.val_loader)

    @torch.no_grad()
    def visualize_predictions(self):
        self.model.eval()

        batch = next(iter(self.val_loader))
        inputs, targets = self._receive_images(batch)

        n_imgs = min(4, inputs.shape[0])
        input_img = inputs[:n_imgs]
        target_img = targets[:n_imgs]
        predicted_ab = self.model(input_img)

        real_rgb = self._lab_to_rgb(input_img, target_img)
        fake_rgb = self._lab_to_rgb(input_img, predicted_ab)
        gray_rgb = ((input_img + 1.0) / 2.0).repeat(1, 3, 1, 1)

        stacked_images = []
        for i in range(n_imgs):
            stacked_images.extend([gray_rgb[i], fake_rgb[i], real_rgb[i]])
            
        stacked_tensor = torch.stack(stacked_images)
        grid = torchvision.utils.make_grid(stacked_tensor, nrow=3, padding=2)

        img_np = grid.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        # Оновлюємо дані на графіку
        if self.img_plot is None:
            self.img_plot = self.ax_img.imshow(img_np)
        else:
            self.img_plot.set_data(img_np)

    def update_metrics_plot(self):
        epochs = list(range(1, len(self.train_loss_history) + 1))

        self.train_line.set_data(epochs, self.train_loss_history)
        if self.val_loss_history:
            self.val_line.set_data(epochs, self.val_loss_history)

        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

    def fit(self, epochs, start_epoch=1, best_val_loss=float('inf')):
        print(f"[INFO] Do model save: {self.do_save}")
        print(f"[INFO] Training starts on {self.device}...")
        self.best_val_loss = best_val_loss

        do_val = bool(self.val_loader)
        length = len(str(epochs))

        try:
            for epoch in range(start_epoch, epochs + 1):
                epoch_str = f"{str(epoch).zfill(length)}/{epochs}"
                
                train_loss = self.train_epoch(epoch_str)
                self.train_loss_history.append(train_loss)

                print(f"[Epoch] {epoch_str} Summary:")
                print(f"[Train] Loss: {train_loss:.4f}")

                save_dict = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                
                if do_val:
                    val_loss = self.validate_epoch(epoch_str)
                    self.val_loss_history.append(val_loss)
                    print(f"[Valid] Loss: {val_loss:.4f}")
                    
                    self.visualize_predictions()

                    if self.do_save:
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            save_path = os.path.join(self.save_dir, f"best_{self.save_name}.pth")
                            save_dict["val_loss"] = val_loss
                            torch.save(save_dict, save_path)
                            print(f"[SAVE] Best model saved: {save_path}")

                self.update_metrics_plot()
                
                if self.do_save:
                    plot_path = os.path.join(self.plot_dir, f"progress_{self.save_name}.png")
                    self.fig.savefig(plot_path, bbox_inches='tight')
                    torch.save(save_dict, os.path.join(self.save_dir, f"latest_{self.save_name}.pth"))

        finally:
            plt.close(self.fig)
            print("[INFO] Training finished. Plot resources released.")