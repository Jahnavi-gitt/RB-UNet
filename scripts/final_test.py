"""Sacred test set final evaluation protocol.

Runs final evaluation on the isolated test split exactly once after models are frozen.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.isic_dataset import ISICDataset
from src.datasets.split_discovery import discover_dataset_splits
from src.datasets.transforms import get_transforms
from src.evaluation.evaluator import Evaluator
from src.models.rb_unet import RBUNet
from src.models.unet import UNet
from src.training.checkpoint import load_checkpoint


def run_final_test(
    baseline_ckpt: str = "outputs/checkpoints/baseline_unet_latest.pth",
    rb_ckpt: str = "outputs/checkpoints/full_rb_unet_latest.pth",
    config_path: str = "configs/config.yaml",
    smoke_test: bool = False,
):
    print("=" * 65)
    print("SACRED FINAL TEST EVALUATION PROTOCOL")
    print("=" * 65)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = os.environ.get("DATASET_ROOT") or cfg.get("dataset", {}).get("root", "data/raw")
    splits = discover_dataset_splits(dataset_root)

    test_samples = splits.get("test", [])
    if len(test_samples) == 0:
        if smoke_test:
            smoke_dir = Path("outputs/smoke_test_data")
            test_samples = [
                {"id": f.stem.replace("_mask", ""), "image_path": str(f.parent / f"{f.stem.replace('_mask', '')}.jpg"), "mask_path": str(f)}
                for f in smoke_dir.glob("*_mask.png")
            ]
        else:
            print("[Final Test] Notice: No labeled test pairs discovered in dataset root.")
            return None

    print(f"[Final Test] Evaluating on {len(test_samples)} sacred test samples...")

    img_size = 128 if smoke_test else cfg.get("dataset", {}).get("image_size", 256)
    test_tf = get_transforms(image_size=img_size, is_training=False)
    test_ds = ISICDataset(test_samples, transform=test_tf)
    loader = DataLoader(test_ds, batch_size=2 if smoke_test else 4, shuffle=False, num_workers=0)

    features = [16, 32, 64, 128, 256] if smoke_test else cfg.get("model", {}).get("features", [16, 32, 64, 128, 256])
    bnd_channels = 16 if smoke_test else cfg.get("model", {}).get("boundary_head_channels", 16)

    from src.training.checkpoint import build_model_from_checkpoint

    # 1. Baseline Evaluation
    if os.path.exists(baseline_ckpt):
        base_model = build_model_from_checkpoint(baseline_ckpt, device=str(device))
    else:
        base_model = UNet(in_channels=3, out_channels=1, features=features).to(device)
    base_eval = Evaluator(base_model, device=device).evaluate(loader, desc="Baseline Test")

    # 2. RB-UNet Evaluation
    if os.path.exists(rb_ckpt):
        rb_model = build_model_from_checkpoint(rb_ckpt, device=str(device))
    else:
        rb_model = RBUNet(in_channels=3, out_channels=1, features=features, boundary_head_channels=bnd_channels).to(device)
    rb_eval = Evaluator(rb_model, device=device).evaluate(loader, desc="RB-UNet Test")

    results = {
        "dataset_root": str(dataset_root),
        "test_sample_count": len(test_samples),
        "baseline_unet": {
            "test_dice": base_eval["mean_dice"],
            "test_iou": base_eval["mean_iou"],
        },
        "rb_unet": {
            "test_dice": rb_eval["mean_dice"],
            "test_iou": rb_eval["mean_iou"],
        },
        "delta_dice": rb_eval["mean_dice"] - base_eval["mean_dice"],
        "delta_iou": rb_eval["mean_iou"] - base_eval["mean_iou"],
    }

    print("\n" + "-" * 55)
    print(f"{'Model':<25} | {'Test Dice':<12} | {'Test IoU':<12}")
    print("-" * 55)
    print(f"{'Standard U-Net Baseline':<25} | {base_eval['mean_dice']:<12.4f} | {base_eval['mean_iou']:<12.4f}")
    print(f"{'RB-UNet (Proposed)':<25} | {rb_eval['mean_dice']:<12.4f} | {rb_eval['mean_iou']:<12.4f}")
    print(f"{'Difference (Delta)':<25} | {results['delta_dice']:>+12.4f} | {results['delta_iou']:>+12.4f}")
    print("-" * 55)

    out_file = Path("outputs/metrics/final_test_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sacred final test evaluation.")
    parser.add_argument("--baseline", type=str, default="outputs/checkpoints/baseline_unet_latest.pth")
    parser.add_argument("--rb-unet", type=str, default="outputs/checkpoints/full_rb_unet_latest.pth")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run_final_test(
        baseline_ckpt=args.baseline,
        rb_ckpt=args.rb_unet,
        config_path=args.config,
        smoke_test=args.smoke_test,
    )
