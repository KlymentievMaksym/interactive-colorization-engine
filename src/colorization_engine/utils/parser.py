import argparse
import torch

from dataclasses import dataclass

def parse_unknown_args(unknown_args: list[str]):
    model_params = {}
    if unknown_args:
        for i in range(0, len(unknown_args), 2):
            key = unknown_args[i].lstrip("--")

            val_str = unknown_args[i+1]
            try:
                val = int(val_str) if val_str.isdigit() else float(val_str)
            except ValueError:
                val = val_str

            model_params[key] = val
    return model_params

@dataclass
class InfoConfig:
    model: str
    weights: str | None
    config: str | None

@dataclass
class DataloaderConfig:
    image_size: int | None

    data: list[str]
    val_data: list[str] | None
    batch_size: int | None

@dataclass
class TrainConfig(DataloaderConfig, InfoConfig):
    device: str

    resume: str | None
    no_save: bool

    epochs: int | None
    lr: float | None

@dataclass
class InferenceConfig(InfoConfig):
    device: str

    image: str
    result: str

    image_size: int | None

@dataclass
class EvalConfig(DataloaderConfig, InfoConfig):
    device: str

    output_dir: str
    save_number: int


class Parser:
    @staticmethod
    def _args_device(parser: argparse.ArgumentParser):
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _args_info(parser: argparse.ArgumentParser):
        parser.add_argument("--model", type=str, required=True)
        parser.add_argument("--weights", type=str, default=None)
        parser.add_argument("--config", type=str, default=None)
        parser.add_argument("--image_size", type=int, default=None)

    @staticmethod
    def _args_dataloader(parser: argparse.ArgumentParser):
        parser.add_argument("--data", type=str, nargs='+', required=True)
        parser.add_argument("--val_data", type=str, nargs='+', default=None)
        parser.add_argument("--batch_size", type=int, default=None)

    @staticmethod
    def _args_hyperparameters_train(parser: argparse.ArgumentParser):
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--lr", type=float, default=None)


    @staticmethod
    def train_args():
        parser = argparse.ArgumentParser(description="Model training")
        parser.add_argument("--resume", type=str, default=None)
        parser.add_argument("--no-save", action="store_false")
        Parser._args_device(parser)
        Parser._args_info(parser)
        Parser._args_dataloader(parser)
        Parser._args_hyperparameters_train(parser)
        return parser.parse_known_args()

    @staticmethod
    def inference_args():
        parser = argparse.ArgumentParser(description="Model inference")
        parser.add_argument("--image", type=str, required=True)
        parser.add_argument("--result", type=str, default=None)  #"result.jpg"
        Parser._args_device(parser)
        Parser._args_info(parser)
        return parser.parse_known_args()
    
    @staticmethod
    def evaluate_args():
        parser = argparse.ArgumentParser(description="Model evaluation")
        parser.add_argument("--output_dir", type=str, default="results/eval")
        parser.add_argument("--save_number", type=int, default=50)
        Parser._args_device(parser)
        Parser._args_info(parser)
        Parser._args_dataloader(parser)
        return parser.parse_known_args()
