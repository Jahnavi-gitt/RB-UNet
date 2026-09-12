"""Unified evaluation script for segmentation models."""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.datasets.isic_dataset import ISICDataset
from src.datasets.split_discovery import discover_dataset_splits
from src.datasets.transforms import get_transforms
from src.evaluation.evaluator import Evaluator
from src.models.rb_unet import RBUNet
from src.models.unet import UNet
from src.training.checkpoint import load_checkpoint


def run_evaluation(checkpoint_path: str, split: str = "val", config_path: str = "configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = os.environ.get("DATASET_ROOT") or cfg.get("dataset", {}).get("root", "data/raw")
    splits = discover_dataset_splits(dataset_root)

    samples = splits.get(split, [])
    if len(samples) == 0:
        print(f"[Evaluation] Warning: No samples found for split '{split}' in {dataset_root}.")
        return None

    img_size = cfg.get("dataset", {}).get("image_size", 256)
    val_tf = get_transforms(image_size=img_size, is_training=False)
    dataset = ISICDataset(samples, transform=val_tf)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    # Detect model type from checkpoint or config
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    is_rb = "boundary_head.head.0.weight" in ckpt["model_state_dict"]

    if is_rb:
        model = RBUNet(in_channels=3, out_channels=1, features=cfg.get("model", {}).get("features", [64, 128, 256, 512, 1024]))
    else:
        model = UNet(in_channels=3, out_channels=1, features=cfg.get("model", {}).get("features", [64, 128, 256, 512, 1024]))

    load_checkpoint(checkpoint_path, model, device=str(device))

    evaluator = Evaluator(model, device=device, threshold=cfg.get("evaluation", {}).get("threshold", 0.5))
    metrics = evaluator.evaluate(loader, desc=f"Evaluating on {split}")

    print(f"\n[Evaluation] Split: {split} | Dice: {metrics['mean_dice']:.4f} | IoU: {metrics['mean_iou']:.4f}")

    out_file = Path("outputs/metrics") / f"eval_{split}_{Path(checkpoint_path).stem}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    run_evaluation(checkpoint_path=args.checkpoint, split=args.split, config_path=args.config)
