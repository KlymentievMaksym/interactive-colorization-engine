from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from colorization_engine.utils.saliency import FastSaliencySampler
from colorization_engine.utils.patches import get_gaussian_patch_circle as get_gaussian_patch

SALIENCY_SAMPLER = FastSaliencySampler(blur_kernel_size=15, uniform_prior=0.15)

class BaseColorizer(nn.Module, ABC):
    """
    Абстрактний базовий клас для всіх моделей колоризації.
    Забезпечує єдиний інтерфейс для інференсу та валідації.
    """
    @abstractmethod
    def forward(self, l_channel: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        """
        l_channel: Tensor форми [B, 1, H, W] [-1, 1] (L)
        hints: Tensor форми [B, 3, H, W] (ab + mask) or None
        Returns: Tensor форми [B, 2, H, W] [-1, 1] (ab)
        """
        pass

    def _generate_random_color(self, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Генерує випадкові кольори у просторі CIELAB з фокусом на природні/приглушені тони.
        """
        # std_dev контролює насиченість. 
        # 0.3-0.4 - це хороше значення для природних зображень
        std_dev = 0.35 
        
        # Генеруємо нормальний розподіл (центр у 0)
        color_ab = torch.randn((2, 1, 1), device=device) * std_dev
        
        # Обрізаємо жорсткі викиди, щоб не вийти за межі [-1, 1]
        color_ab = torch.clamp(color_ab, min=-1.0, max=1.0)
        
        return color_ab

    def _generate_random_gaussian_hints(self, l_channel: torch.Tensor, base_hints: torch.Tensor | None, num_hints: int = 5, min_hint_size: int = 2, max_hint_size: int = 16) -> torch.Tensor:
        """
        Генерує тензор випадкових підказок у незайнятих місцях.
        """
        B, _, H, W = l_channel.shape
        device = l_channel.device

        # Канали: [a, b, mask]
        new_hints = torch.zeros((B, 3, H, W), device=device)

        if base_hints is not None:
            # Вважаємо зайнятим все, що має маску > 0.05
            occupied = base_hints[:, 2:3, :, :] > 0.05 
        else:
            occupied = torch.zeros((B, 1, H, W), dtype=torch.bool, device=device)

        min_dist_sq = (max_hint_size * 1.5) ** 2 

        for b in range(B):
            if num_hints <= 0:
                continue

            points = SALIENCY_SAMPLER.sample_points(l_channel[b:b+1], num_hints * 3)
            
            if not points:
                continue

            placed_points = []

            for y, x in points:
                if occupied[b, 0, y, x]:
                    continue

                too_close = any((y - py)**2 + (x - px)**2 < min_dist_sq for py, px in placed_points)
                if too_close:
                    continue

                placed_points.append((y, x))

                pad = int(torch.randint(min_hint_size, max_hint_size, (1,)).item())
                patch_size = pad * 2 + 1
                base_patch = get_gaussian_patch(patch_size, device=device) # type: ignore

                intensity = torch.rand(1, device=device).item() * 0.8 + 0.2
                patch = base_patch * intensity

                y1, y2 = max(0, y - pad), min(H, y + pad + 1)
                x1, x2 = max(0, x - pad), min(W, x + pad + 1)

                py1, py2 = max(0, pad - y), patch_size - max(0, y + pad + 1 - H)
                px1, px2 = max(0, pad - x), patch_size - max(0, x + pad + 1 - W)

                current_mask_patch = patch[py1:py2, px1:px2]
                new_hints[b, 2, y1:y2, x1:x2] = torch.maximum(new_hints[b, 2, y1:y2, x1:x2], current_mask_patch)

                random_color = self._generate_random_color(device=device)
                colored_patch = random_color * current_mask_patch
                new_hints[b, 0:2, y1:y2, x1:x2] = new_hints[b, 0:2, y1:y2, x1:x2] + colored_patch

                if len(placed_points) >= num_hints:
                    break

        return new_hints

    @torch.no_grad()
    def sample(self, l_channel: torch.Tensor, hints: torch.Tensor | None = None, num_samples: int = 5, color_intensity: float = 1.0, min_hint_size: int = 2, max_hint_size: int = 8) -> list[torch.Tensor]:
        """
        Генерує варіанти колоризації, додаючи випадкові кольорові плями в порожні зони.
        """
        self.eval()
        variants = []

        clean_pred = self(l_channel, hints)
        variants.append(clean_pred)

        for _ in range(num_samples - 1):
            gen_hints = self._generate_random_gaussian_hints(l_channel=l_channel, base_hints=hints, min_hint_size=min_hint_size, max_hint_size=max_hint_size)
            gen_hints[:, :2, :, :] *= color_intensity
            gen_hints[:, :2, :, :] = torch.clamp(gen_hints[:, :2, :, :], -1.0, 1.0)

            if hints is not None:
                user_color = hints[:, :2, :, :]
                user_mask = hints[:, 2:3, :, :]
                
                gen_color = gen_hints[:, :2, :, :]
                gen_mask = gen_hints[:, 2:3, :, :]

                combined_mask = torch.maximum(user_mask, gen_mask)

                combined_color = user_color + gen_color
                combined_color = torch.clamp(combined_color, -1.0, 1.0)
                # is_user = user_mask > 0
                # combined_color = torch.where(is_user, user_color, gen_color)

                combined_hints = torch.cat([combined_color, combined_mask], dim=1)
            else:
                combined_hints = gen_hints

            variant_pred = self(l_channel, combined_hints)
            variants.append(variant_pred)
            
        return variants