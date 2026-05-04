from types import FunctionType

MODEL_REGISTRY = {}
LOSS_REGISTRY = {}

def register_model(name: str) -> FunctionType:
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper

def register_loss(name: str) -> FunctionType:
    def wrapper(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return wrapper