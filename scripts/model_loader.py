import torch
from pathlib import Path

from colorization_engine.models import MODEL_REGISTRY

def load_colorization_model(model_name: str, device: torch.device, checkpoint_path: str | Path | None = None, **override_params) -> torch.nn.Module:
    """
    Фабрика для ініціалізації будь-якої моделі з єдиного реєстру.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Модель {model_name} не знайдена в реєстрі. Доступні: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[model_name]
    weights_path = Path(checkpoint_path) if checkpoint_path else Path(config["weights_path"])

    if not weights_path.exists():
        raise FileNotFoundError(f"[ERROR] Ваги не знайдені за шляхом: {weights_path}. Завантажте їх спочатку.")

    model_wrapper = config["class"](**config["params"], **override_params)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model_wrapper.model.load_state_dict(state_dict)

    model_wrapper = model_wrapper.to(device)
    model_wrapper.eval()

    return model_wrapper