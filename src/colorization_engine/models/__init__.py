from colorization_engine.models.zhang import ZhangWrapper
from colorization_engine.models.mamba import MambaWrapper
from colorization_engine.models.mamba2 import MambaWrapper2
from colorization_engine.models.deoldify import DeOldifyWrapper

MODEL_REGISTRY = {
    "mamba": {
        "class": MambaWrapper,
        "weights_path": "checkpoints/mamba.pth",
        "params": {
            "d_model": 256,
            "layers": 6,
            "blocks": 2
        },
    },
    "mamba2": {
        "class": MambaWrapper2,
        "weights_path": "checkpoints/mamba2.pth",
        "params": {
            "d_model": 256,
            "layers": 6,
            "blocks": 2
        },
    },
    "zhang_siggraph17": {
        "class": ZhangWrapper,
        "weights_path": "checkpoints/siggraph17-v1.pth",
        "params": {},
    },
    "deoldify_artistic": {
        "class": DeOldifyWrapper,
        "weights_path": "checkpoints/ColorizeArtistic_gen.pth",
        "params": {},
    }
}