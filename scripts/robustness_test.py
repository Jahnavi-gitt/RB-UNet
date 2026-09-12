"""Benchmark baseline U-Net vs RB-UNet under controlled image degradations."""

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
from src.evaluation.robustness import RobustnessBenchmark
from src.models.rb_unet import RBUNet
from src.models.unet import UNet
from src.training.checkpoint import load_checkpoint


def run_robustness_test(
    baseline_ckpt: str = "outputs/checkpoints/baseline_unet_latest.pth",
    rb_ckpt: str = "outputs/checkpoints/full_rb_unet_latest.pth",
    config_path: str = "configs/config.yaml",
    smoke_test: bool = False,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = os.environ.get("DATASET_ROOT") or cfg.get("dataset", {}).get("root", "data/raw")
    splits = discover_dataset_splits(dataset_root)

    val_samples = splits.get("val", [])
    if smoke_test or len(val_samples) == 0:
        print("[Robustness] Notice: Using smoke test samples for robustness benchmark.")
        smoke_dir = Path("outputs/smoke_test_data")
        val_samples = [
            {"id": f.stem.replace("_mask", ""), "image_path": str(f.parent / f"{f.stem.replace('_mask', '')}.jpg"), "mask_path": str(f)}
            for f in smoke_dir.glob("*_mask.png")
        ]

    val_tf = get_transforms(image_size=128 if smoke_test else cfg.get("dataset", {}).get("image_size", 256), is_training=False)
    dataset = ISICDataset(val_samples, transform=val_tf)
    loader = DataLoader(dataset, batch_size=2 if smoke_test else 4, shuffle=False, num_workers=0)

    from src.training.checkpoint import build_model_from_checkpoint

    if os.path.exists(baseline_ckpt):
        base_model = build_model_from_checkpoint(baseline_ckpt, device=str(device))
    else:
        base_model = UNet(in_channels=3, out_channels=1, features=features).to(device)

    if os.path.exists(rb_ckpt):
        rb_model = build_model_from_checkpoint(rb_ckpt, device=str(device))
    else:
        bnd_channels = cfg.get("model", {}).get("boundary_head_channels", 16)
        rb_model = RBUNet(in_channels=3, out_channels=1, features=features, boundary_head_channels=bnd_channels).to(device)

    benchmark = RobustnessBenchmark(baseline_model=base_model, rb_unet_model=rb_model, device=device)
    results = benchmark.run_benchmark(loader)

    # Save outputs
    out_dir = Path("outputs/robustness")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "robustness_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Write Markdown table
    md_path = out_dir / "robustness_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Robustness Evaluation Benchmark\n\n")
        f.write("| Condition | Baseline U-Net Dice | RB-UNet Dice | Delta Dice | Baseline IoU | RB-UNet IoU | Delta IoU |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            f.write(
                f"| **{r['condition']}** | {r['baseline_dice']:.4f} | {r['rb_unet_dice']:.4f} | "
                f"**{r['delta_dice']:+.4f}** | {r['baseline_iou']:.4f} | {r['rb_unet_iou']:.4f} | "
                f"{r['delta_iou']:+.4f} |\n"
            )

    print(f"\n[Robustness] Results saved to:\n  - {out_dir / 'robustness_results.json'}\n  - {md_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate robustness under image corruptions.")
    parser.add_argument("--baseline", type=str, default="outputs/checkpoints/baseline_unet_latest.pth")
    parser.add_argument("--rb-unet", type=str, default="outputs/checkpoints/full_rb_unet_latest.pth")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run_robustness_test(
        baseline_ckpt=args.baseline,
        rb_ckpt=args.rb_unet,
        config_path=args.config,
        smoke_test=args.smoke_test,
    )
