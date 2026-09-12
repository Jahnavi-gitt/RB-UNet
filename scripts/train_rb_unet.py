"""Experiment 4: Train Full Proposed RB-UNet (Boundary + Consistency)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Full Proposed RB-UNet (Exp 4)")
    parser.add_argument("--config", type=str, default="configs/rb_unet.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        experiment_name="full_rb_unet",
        smoke_test=args.smoke_test,
        override_epochs=args.epochs,
    )
