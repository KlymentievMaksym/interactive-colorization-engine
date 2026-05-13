from abc import ABC, abstractmethod
import torch
import torch.nn as nn

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

    def _generate_random_gaussian_hints(
        self, 
        base_hints: torch.Tensor | None, 
        B: int, H: int, W: int, 
        device: torch.device,
        num_hints: int = 5,
        radius: float = 4.0
    ) -> torch.Tensor:
        """
        Генерує тензор випадкових підказок у незайнятих місцях.
        """
        new_hints = torch.zeros((B, 3, H, W), device=device)
        
        # Визначаємо зайняту територію (де маска користувача > 0)
        if base_hints is not None:
            # Вважаємо зайнятим все, що має маску > 0.05
            occupied = base_hints[:, 2:3, :, :] > 0.05 
        else:
            occupied = torch.zeros((B, 1, H, W), dtype=torch.bool, device=device)

        # Створюємо координатні сітки для швидкого малювання Гаусса
        y_grid = torch.arange(H, device=device).view(-1, 1).float()
        x_grid = torch.arange(W, device=device).view(1, -1).float()

        for b in range(B):
            # Знаходимо всі координати (y, x), які НЕ зайняті
            free_space = ~occupied[b, 0]
            valid_coords = torch.nonzero(free_space) # [N, 2]
            
            if len(valid_coords) == 0:
                continue # Якщо користувач замалював усе, пропускаємо

            # Генеруємо задану кількість випадкових точок
            for _ in range(num_hints):
                # 1. Вибираємо випадкову вільну координату
                idx = torch.randint(0, len(valid_coords), (1,)).item()
                cy, cx = valid_coords[idx]

                # 2. Генеруємо випадковий колір a та b в діапазоні [-1, 1]
                rand_a = (torch.rand(1, device=device).item() * 2.0) - 1.0
                rand_b = (torch.rand(1, device=device).item() * 2.0) - 1.0

                # 3. Малюємо гауссову пляму
                dist_sq = (y_grid - cy)**2 + (x_grid - cx)**2
                sigma = radius / 3.0 # Щоб на межі радіуса значення падало майже до 0
                patch_mask = torch.exp(-dist_sq / (2 * sigma**2))
                
                # Обрізаємо хвости Гаусса, щоб пляма була локальною
                patch_mask[dist_sq > radius**2] = 0.0 
                
                # Забороняємо плямі залізати на територію користувача
                patch_mask[occupied[b, 0]] = 0.0
                
                patch_mask = patch_mask.unsqueeze(0) # [1, H, W]

                # 4. Додаємо пляму до загального тензора нових підказок
                # Використовуємо маску для оновлення кольору там, де нова пляма сильніша
                update_mask = patch_mask > new_hints[b, 2:3]
                new_hints[b, 0:1][update_mask] = rand_a
                new_hints[b, 1:2][update_mask] = rand_b
                new_hints[b, 2:3] = torch.maximum(new_hints[b, 2:3], patch_mask)

        return new_hints

    @torch.no_grad()
    def sample(
        self, 
        l_channel: torch.Tensor, 
        hints: torch.Tensor | None = None, 
        num_samples: int = 5, 
        random_hints_count: int = 4,
        patch_radius: float = 3.0
    ) -> list[torch.Tensor]:
        """
        Генерує варіанти колоризації, додаючи випадкові кольорові плями в порожні зони.
        """
        self.eval()
        variants = []
        B, _, H, W = l_channel.shape
        device = l_channel.device

        # Варіант 1: Точно те, що попросив користувач (чистий предикт)
        clean_pred = self(l_channel, hints)
        variants.append(clean_pred)

        # Решта варіантів: генеруємо випадкові підказки
        for _ in range(num_samples - 1):
            # Генеруємо нові випадкові підказки
            gen_hints = self._generate_random_gaussian_hints(
                base_hints=hints, 
                B=B, H=H, W=W, 
                device=device,
                num_hints=random_hints_count,
                radius=patch_radius
            )

            # Об'єднуємо підказки користувача та згенеровані
            if hints is not None:
                user_color = hints[:, :2, :, :]
                user_mask = hints[:, 2:3, :, :]
                
                gen_color = gen_hints[:, :2, :, :]
                gen_mask = gen_hints[:, 2:3, :, :]

                # Фінальна маска — це максимум з двох
                combined_mask = torch.maximum(user_mask, gen_mask)
                
                # Колір: пріоритет у користувача. 
                # Якщо маска юзера > 0, беремо його колір, інакше згенерований
                is_user = user_mask > 0
                combined_color = torch.where(is_user, user_color, gen_color)
                
                combined_hints = torch.cat([combined_color, combined_mask], dim=1)
            else:
                combined_hints = gen_hints

            # Робимо інференс з об'єднаними підказками
            variant_pred = self(l_channel, combined_hints)
            variants.append(variant_pred)
            
        return variants