"""Unified training entrypoint for all RB-UNet ablation experiments."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.datasets.isic_dataset import ISICDataset
from src.datasets.split_discovery import discover_dataset_splits
from src.datasets.transforms import get_transforms
from src.models.rb_unet import RBUNet
from src.models.unet import UNet
from src.training.seed import set_seed
from src.training.trainer import Trainer
from src.utils.logging import log_training_history, save_json_metrics
from src.utils.visualization import plot_training_curves


def run_training(
    config_path: str = "configs/config.yaml",
    experiment_name: Optional[str] = None,
    smoke_test: bool = False,
    override_epochs: Optional[int] = None,
):
    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Base config fallback if this is a sub-config
    base_config_path = "configs/config.yaml"
    if config_path != base_config_path and os.path.exists(base_config_path):
        with open(base_config_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
            # Merge configs
            merged = base_cfg.copy()
            for k, v in cfg.items():
                if isinstance(v, dict) and k in merged:
                    merged[k].update(v)
                else:
                    merged[k] = v
            cfg = merged

    # Set reproducibility
    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    # Device
    dev_str = cfg.get("training", {}).get("device", "auto")
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)
    print(f"[Run] Using computational device: {device}")

    # Dataset discovery
    dataset_root = os.environ.get("DATASET_ROOT") or cfg.get("dataset", {}).get("root", "data/raw")
    discovered = discover_dataset_splits(dataset_root)

    train_samples = discovered["train"]
    val_samples = discovered["val"]

    # Fallback for smoke test or missing dataset
    if smoke_test or len(train_samples) == 0:
        print("[Run] Notice: Running in smoke-test / dry-run mode using synthesized data fixtures.")
        smoke_dir = Path("outputs/smoke_test_data")
        smoke_dir.mkdir(parents=True, exist_ok=True)
        train_samples = []
        val_samples = []
        for i in range(8):
            sid = f"SYNTH_{i:03d}"
            img_path = smoke_dir / f"{sid}.jpg"
            msk_path = smoke_dir / f"{sid}_mask.png"
            if not img_path.exists():
                import numpy as np
                from PIL import Image
                img = np.full((128, 128, 3), 160, dtype=np.uint8)
                msk = np.zeros((128, 128), dtype=np.uint8)
                y, x = np.ogrid[:128, :128]
                mask_area = (x - 64) ** 2 + (y - 64) ** 2 <= 28 ** 2
                msk[mask_area] = 255
                img[mask_area] = [60, 30, 20]
                Image.fromarray(img).save(img_path)
                Image.fromarray(msk).save(msk_path)

            sample_dict = {"id": sid, "image_path": str(img_path), "mask_path": str(msk_path)}
            if i < 6:
                train_samples.append(sample_dict)
            else:
                val_samples.append(sample_dict)

    print(f"[Run] Loaded {len(train_samples)} training samples, {len(val_samples)} validation samples.")

    # Image transforms
    img_size = cfg.get("dataset", {}).get("image_size", 256)
    if smoke_test:
        img_size = 128

    train_tf = get_transforms(image_size=img_size, is_training=True)
    val_tf = get_transforms(image_size=img_size, is_training=False)

    train_ds = ISICDataset(train_samples, transform=train_tf)
    val_ds = ISICDataset(val_samples, transform=val_tf)

    batch_size = cfg.get("training", {}).get("batch_size", 4)
    if smoke_test:
        batch_size = 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model instantiation
    model_name = cfg.get("model", {}).get("name", "rb_unet")
    features = cfg.get("model", {}).get("features", [64, 128, 256, 512, 1024])
    if smoke_test:
        features = [16, 32, 64, 128, 256]

    if model_name == "standard_unet":
        model = UNet(in_channels=3, out_channels=1, features=features)
    else:
        bnd_channels = cfg.get("model", {}).get("boundary_head_channels", 32)
        if smoke_test:
            bnd_channels = 16
        model = RBUNet(in_channels=3, out_channels=1, features=features, boundary_head_channels=bnd_channels)

    exp_name = experiment_name or model_name
    print(f"[Run] Instantiated architecture: {model.__class__.__name__} ({exp_name})")

    # Optimizer
    lr = cfg.get("training", {}).get("learning_rate", 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = override_epochs or (2 if smoke_test else cfg.get("training", {}).get("epochs", 25))

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=cfg,
        device=device,
        checkpoint_dir="outputs/checkpoints",
        experiment_name=exp_name,
    )

    result = trainer.fit(epochs=epochs)

    # Save training curves and metrics
    metrics_dir = Path("outputs/metrics")
    plots_dir = Path("outputs/plots")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_training_history(result["history"], str(metrics_dir / f"{exp_name}_history"))
    plot_training_curves(result["history"], str(plots_dir / f"{exp_name}_convergence.png"))

    summary = {
        "experiment": exp_name,
        "model": model.__class__.__name__,
        "best_val_dice": result["best_val_dice"],
        "epochs": epochs,
        "elapsed_seconds": result["elapsed_seconds"],
        "checkpoint": f"outputs/checkpoints/{exp_name}_latest.pth",
    }
    save_json_metrics(summary, str(metrics_dir / f"{exp_name}_summary.json"))

    print(f"[Run] Experiment '{exp_name}' finished. Best Val Dice: {result['best_val_dice']:.4f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RB-UNet models.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        experiment_name=args.name,
        smoke_test=args.smoke_test,
        override_epochs=args.epochs,
    )
