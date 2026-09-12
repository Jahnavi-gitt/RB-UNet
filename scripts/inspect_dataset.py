"""Standalone dataset audit and split discovery script for RB-UNet.

Inspects the dataset root, checks image-mask pair integrity, reports dimensions,
and outputs a structured audit report to console and outputs/metrics/dataset_report.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from PIL import Image
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.split_discovery import discover_dataset_splits


def inspect_dataset(config_path: str = "configs/config.yaml", custom_root: str = None) -> dict:
    # Load configuration
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    dataset_root = custom_root or os.environ.get("DATASET_ROOT") or cfg.get("dataset", {}).get("root", "data/raw")

    print("=" * 65)
    print("ISIC 2018 DATASET AUDIT & SPLIT DISCOVERY")
    print("=" * 65)
    print(f"Configured Dataset Root: {dataset_root}")

    discovery_result = discover_dataset_splits(dataset_root)
    stats = discovery_result["stats"]

    print("\n--- Split Discovery Results ---")
    print(f"Directory Exists:      {discovery_result['exists']}")
    print(f"Training Pairs:        {stats.get('total_train', 0)}")
    print(f"Validation Pairs:      {stats.get('total_val', 0)}")
    print(f"Test Pairs:            {stats.get('total_test', 0)}")
    print(f"Train Image Dir:       {stats.get('train_image_dir')}")
    print(f"Train Mask Dir:        {stats.get('train_mask_dir')}")
    print(f"Val Image Dir:         {stats.get('val_image_dir')}")
    print(f"Test Image Dir:        {stats.get('test_image_dir')}")
    print(f"Train Orphaned Images: {stats.get('train_orphans_count', 0)}")

    # Check sample dimensions if samples exist
    sample_info = {}
    if discovery_result["train"]:
        sample = discovery_result["train"][0]
        try:
            with Image.open(sample["image_path"]) as img:
                sample_info["image_dimensions"] = list(img.size)
                sample_info["image_mode"] = img.mode
            with Image.open(sample["mask_path"]) as msk:
                sample_info["mask_dimensions"] = list(msk.size)
                sample_info["mask_mode"] = msk.mode
            print(f"\nSample Inspection ({sample['id']}):")
            print(f"  Image Size: {sample_info['image_dimensions']} ({sample_info['image_mode']})")
            print(f"  Mask Size:  {sample_info['mask_dimensions']} ({sample_info['mask_mode']})")
        except Exception as e:
            print(f"  Warning: Could not read sample: {e}")

    # Prepare report dict
    report = {
        "dataset_root": str(dataset_root),
        "exists": discovery_result["exists"],
        "splits": {
            "train_count": stats.get("total_train", 0),
            "val_count": stats.get("total_val", 0),
            "test_count": stats.get("total_test", 0),
        },
        "directories": {
            "train_images": stats.get("train_image_dir"),
            "train_masks": stats.get("train_mask_dir"),
            "val_images": stats.get("val_image_dir"),
            "val_masks": stats.get("val_mask_dir"),
            "test_images": stats.get("test_image_dir"),
            "test_masks": stats.get("test_mask_dir"),
        },
        "orphans": {
            "train": stats.get("train_orphans_count", 0),
            "val": stats.get("val_orphans_count", 0),
            "test": stats.get("test_orphans_count", 0),
        },
        "sample_properties": sample_info,
    }

    # Save to outputs/metrics/dataset_report.json
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_file = metrics_dir / "dataset_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nAudit report saved to: {report_file}")
    print("=" * 65)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and inspect ISIC dataset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--root", type=str, default=None, help="Override dataset root path")
    args = parser.parse_args()

    inspect_dataset(config_path=args.config, custom_root=args.root)
