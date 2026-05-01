import torch
import torch.nn.functional as F

class FastSaliencySampler:
    """
    Клас для генерації координат підказок на основі градієнтної значущості.
    Оптимізовано для використання у PyTorch DataLoader.
    """
    def __init__(self, blur_kernel_size: int = 15, uniform_prior: float = 0.1):
        """
        Args:
            blur_kernel_size: Розмір вікна для розмиття градієнтів (чим більше, тим ширша зона навколо країв).
            uniform_prior: Вага рівномірного розподілу [0, 1]. Забезпечує ймовірність потрапляння на фон.
        """
        self.kernel_size = blur_kernel_size
        self.uniform_prior = uniform_prior
        
        # Ініціалізація ядер Собеля (1, 1, 3, 3)
        sobel_x = torch.tensor([[-1., 0., 1.], 
                                [-2., 0., 2.], 
                                [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], 
                                [0., 0., 0.], 
                                [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Реєструємо як буфери, щоб вони не вимагали градієнтів
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    def get_pdf(self, l_tensor: torch.Tensor) -> torch.Tensor:
        """Обчислює 2D функцію щільності ймовірності (PDF) для зображення."""
        # Підготовка розмірностей: [C, H, W] -> [1, 1, H, W]
        x = l_tensor.unsqueeze(0) if l_tensor.dim() == 3 else l_tensor
        
        # 1. Обчислення градієнтів
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        
        # 2. Магнітуда
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
        
        # 3. Розмиття (поширення значущості від країв усередину об'єктів)
        padding = self.kernel_size // 2
        saliency = F.avg_pool2d(magnitude, kernel_size=self.kernel_size, stride=1, padding=padding)
        saliency = saliency.squeeze(0).squeeze(0) # [H, W]
        
        # 4. Нормалізація та змішування з Uniform Prior
        saliency_norm = saliency / (saliency.sum() + 1e-8)
        uniform_dist = torch.ones_like(saliency_norm) / (saliency_norm.numel())
        
        pdf = (1.0 - self.uniform_prior) * saliency_norm + self.uniform_prior * uniform_dist
        return pdf

    def sample_points(self, l_tensor: torch.Tensor, num_points: int) -> list[tuple[int, int]]:
        """
        Повертає список координат (y, x) на основі карти значущості.
        """
        if num_points <= 0:
            return []
            
        pdf = self.get_pdf(l_tensor)
        h, w = pdf.shape
        
        # Flatten PDF для семплювання: [H, W] -> [H*W]
        pdf_flat = pdf.view(-1)
        
        # Семплюємо індекси без повернення (replacement=False)
        sampled_indices = torch.multinomial(pdf_flat, num_points, replacement=False)
        
        # Відновлюємо 2D координати
        points = []
        for idx in sampled_indices:
            y = (idx // w).item()
            x = (idx % w).item()
            points.append((y, x))
            
        return points