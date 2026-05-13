import argparse
from pathlib import Path
import logging

import torch

from colorization_engine.utils.logger import setup_logger

logger = setup_logger("ckpt_optimizer")

def optimize_ckpt():
    parser = argparse.ArgumentParser(description="Strip redundant weights (LPIPS/FID/KID) from Lightning checkpoints.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the original .ckpt file")
    parser.add_argument("--out", type=str, default=None, help="Output path (defaults to [name]_clean.ckpt)")
    parser.add_argument("--verbose", action="store_true", help="Show more detailed logs")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    input_path = Path(args.ckpt)
    output_path = Path(args.out) if args.out else input_path.with_name(f"{input_path.stem}_clean{input_path.suffix}")

    logger.info(f"Analyzing checkpoint: {input_path.name}")
    
    try:
        # Load full checkpoint
        checkpoint = torch.load(input_path, map_location="cpu")

        if "state_dict" not in checkpoint:
            logger.error(f"No state_dict found in {input_path.name}!")
            return

        sd = checkpoint["state_dict"]
        initial_keys = len(sd)

        garbage_prefixes = ["lpips", "fid", "kid", "inception"]
        keys_to_remove = [k for k in sd.keys() if any(p in k.lower() for p in garbage_prefixes)]

        logger.info(f"Total keys found: {initial_keys}")
        logger.info(f"Redundant keys identified: {len(keys_to_remove)}")

        if len(keys_to_remove) == 0:
            logger.warning("No redundant keys found. Checkpoint is likely already clean.")
            return

        for k in keys_to_remove:
            logger.debug(f"Removing key: {k}")
            del sd[k]

        initial_size = input_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"Saving optimized checkpoint to: {output_path.name}...")
        torch.save(checkpoint, output_path)

        final_size = output_path.stat().st_size / (1024 * 1024)
        saved_size = initial_size - final_size

        logger.info("-" * 40)
        logger.info(f"Optimization successful:")
        logger.info(f"  Original Size: {initial_size:.2f} MB")
        logger.info(f"  New Size:      {final_size:.2f} MB")
        logger.info(f"  Space Saved:   {saved_size:.2f} MB")
        logger.info("-" * 40)

    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during optimization: {e}")

if __name__ == "__main__":
    optimize_ckpt()