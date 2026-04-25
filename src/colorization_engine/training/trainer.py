import os
import torch
import matplotlib
matplotlib.use('TkAgg')  # <--- MUST BE BEFORE pyplot IMPORT
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import cv2
from tqdm import tqdm

class ColorizationTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device: str = 'cpu', do_save: bool = True, save_name: str | None = None, save_dir: str = 'checkpoints'):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.do_save = do_save
        self.save_name = save_name if  save_name is not None else "model"
        self.save_dir = save_dir

        plt.ion() 
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.ax.axis('off')
        self.fig.suptitle("Live Validation: [Input L | Prediction ab | Target ab]")
        self.live_plot = None

        if self.do_save:
            os.makedirs(self.save_dir, exist_ok=True)

    def _receive_images(self, batch):
        inputs = batch["input"].to(self.device)
        targets = batch["target"].to(self.device)
        return inputs, targets

    def _receive_loss(self, inputs, targets):
        outputs = self.model(inputs)
        return self.criterion(outputs, targets)

    def _update_loss(self, loss, loss_metrics, batches_pbar):
        self.running_loss += loss.item()
        batches_pbar.set_postfix({"loss": f"{loss.item():.4f}", "metrics": loss_metrics})

    def _lab_to_rgb(self, input_img, target_img):
        _input_img = (input_img + 1.0) * 50.0
        _target_img = target_img * 110.0
        
        # [B, 3, H, W]
        lab_tensor = torch.cat([_input_img, _target_img], dim=1)
        
        B = lab_tensor.shape[0]
        rgb_tensors = []
        
        for b in range(B):
            # [3, H, W] -> [H, W, 3]
            lab_np = lab_tensor[b].permute(1, 2, 0).cpu().numpy().astype(np.float32)
            rgb_np = cv2.cvtColor(lab_np, cv2.COLOR_Lab2RGB)
            
            # [H, W, 3] -> [3, H, W]
            rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
            rgb_tensors.append(rgb_tensor)
            
        # Stack back into a batch: [B, 3, H, W]
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
        input_img = inputs[:n_imgs]   # [N, 1, H, W]
        target_img = targets[:n_imgs] # [N, 2, H, W]
        predicted_ab = self.model(input_img) # [N, 2, H, W]

        # 2. RGB [N, 3, H, W]
        real_rgb = self._lab_to_rgb(input_img, target_img)
        fake_rgb = self._lab_to_rgb(input_img, predicted_ab)
        gray_rgb = ((input_img + 1.0) / 2.0).repeat(1, 3, 1, 1)

        stacked_images = []
        for i in range(n_imgs):
            stacked_images.extend([gray_rgb[i], fake_rgb[i], real_rgb[i]])
            
        # [N*3, 3, H, W]
        stacked_tensor = torch.stack(stacked_images)

        grid = torchvision.utils.make_grid(stacked_tensor, nrow=3, padding=2)

        # [3, H_grid, W_grid] -> [H_grid, W_grid, 3]
        img_np = grid.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        
        if self.live_plot is None:
            self.live_plot = self.ax.imshow(img_np)
        else:
            self.live_plot.set_data(img_np)
            
        plt.pause(0.1)

    def fit(self, epochs, start_epoch=1, best_val_loss=float('inf')):
        print(f"[INFO] Do model save: {self.do_save}")
        print(f"[INFO] Training starts on {self.device}...")
        self.best_val_loss = best_val_loss

        do_val = bool(self.val_loader)

        length = len(str(epochs))
        for epoch in range(start_epoch, epochs + 1):
            epoch_str = f"{str(epoch).zfill(length)}/{epochs}"
            train_loss = self.train_epoch(epoch_str)

            print(f"[Epoch] {epoch_str} Summary:")
            print(f"[Train] Loss: {train_loss}")

            save_dict = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            if do_val:
                val_loss = self.validate_epoch(epoch_str)
                self.visualize_predictions()
                print(f"[Valid] Loss: {val_loss}")

                if self.do_save:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_path = os.path.join(self.save_dir, f"best_{self.save_name}.pth")
                        save_dict["val_loss"] = val_loss
                        torch.save(save_dict, save_path)
                        print(f"[SAVE] Best model saved: {save_path}")

            if self.do_save:
                torch.save(save_dict, os.path.join(self.save_dir, f"latest_{self.save_name}.pth"))