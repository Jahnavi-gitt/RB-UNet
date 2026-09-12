"""Experiment 3: Train U-Net with Corruption Consistency Regularization."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net + Corruption Consistency (Exp 3)")
    parser.add_argument("--config", type=str, default="configs/consistency.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        experiment_name="ablation_consistency",
        smoke_test=args.smoke_test,
        override_epochs=args.epochs,
    )
