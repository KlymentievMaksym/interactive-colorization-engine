import argparse
from pathlib import Path
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from colorization_engine.utils.logger import setup_logger

logger = setup_logger("ckpt_inspector")
console = Console()

def inspect_ckpt():
    parser = argparse.ArgumentParser(description="Inspect PyTorch Lightning checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the .ckpt file")
    parser.add_argument("--verbose", action="store_true", help="List all keys")
    parser.add_argument("--weights_only", action="store_true", help="Load only weights (faster, but no metadata)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        logger.error(f"File not found: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=args.weights_only)

    meta_table = Table(title=f"Checkpoint: {ckpt_path.name}", title_style="red")
    meta_table.add_column("Param", justify="left", style="cyan")
    meta_table.add_column("Value", justify="right")
    meta_table.add_row("Size", f"{ckpt_path.stat().st_size / (1024**2):.2f} MB")
    meta_table.add_row("Epoch", str(checkpoint.get("epoch", "N/A")))
    meta_table.add_row("Global Step", str(checkpoint.get("global_step", "N/A")))

    for key in ["hyper_parameters", "datamodule_hyper_parameters"]:
        data = checkpoint.get(key)
        if data:
            meta_table.add_section()
            meta_table.add_row(f"[bold cyan]{key}[/]", "")
            for k, v in data.items():
                meta_table.add_row(f"- {k}", str(v))

    console.print(Panel(meta_table, expand=False, border_style="green"))

    if "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        console.print(f"[bold]State Dict:[/] {len(sd)} total keys")

        if args.verbose:
            sd_table = Table(title="State Dict Keys", title_justify="left")
            sd_table.add_column("Key", style="dim")
            sd_table.add_column("Shape")
            
            for key in sorted(sd.keys()):
                val = sd[key]
                shape = str(list(val.shape)) if hasattr(val, 'shape') else str(type(val))
                sd_table.add_row(key, shape)
            
            console.print(sd_table)

if __name__ == "__main__":
    inspect_ckpt()