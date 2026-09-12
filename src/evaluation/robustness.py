"""Deterministic robustness benchmark across image degradation conditions."""

from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruption.corruptions import apply_named_corruption
from src.evaluation.metrics import compute_dice, compute_iou


class RobustnessBenchmark:
    """Evaluates and compares model performance under controlled image corruptions."""

    CONDITIONS = [
        ("Clean", {}),
        ("Gaussian Noise", {"std": 0.08}),
        ("Gaussian Blur", {"kernel_size": 5, "sigma": 1.5}),
        ("Contrast Shift", {"factor": 0.6}),
        ("Brightness Shift", {"offset": -0.2}),
    ]

    def __init__(
        self,
        baseline_model: nn.Module,
        rb_unet_model: nn.Module,
        device: torch.device,
        threshold: float = 0.5,
    ):
        self.baseline = baseline_model.to(device).eval()
        self.rb_unet = rb_unet_model.to(device).eval()
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def evaluate_condition(
        self,
        dataloader: DataLoader,
        condition_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, float]:
        baseline_dices = []
        baseline_ious = []
        rb_dices = []
        rb_ious = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Corrupt images deterministically
            if condition_name.lower() != "clean":
                corrupt_images = apply_named_corruption(images, condition_name, params).to(self.device)
            else:
                corrupt_images = images

            # Baseline inference
            out_base = self.baseline(corrupt_images)
            pred_base = out_base["seg_logits"] if isinstance(out_base, dict) else out_base
            prob_base = torch.sigmoid(pred_base)

            # RB-UNet inference
            out_rb = self.rb_unet(corrupt_images)
            pred_rb = out_rb["seg_logits"] if isinstance(out_rb, dict) else out_rb
            prob_rb = torch.sigmoid(pred_rb)

            for b in range(len(images)):
                baseline_dices.append(compute_dice(prob_base[b], masks[b], threshold=self.threshold))
                baseline_ious.append(compute_iou(prob_base[b], masks[b], threshold=self.threshold))
                rb_dices.append(compute_dice(prob_rb[b], masks[b], threshold=self.threshold))
                rb_ious.append(compute_iou(prob_rb[b], masks[b], threshold=self.threshold))

        mean_base_dice = float(np.mean(baseline_dices)) if baseline_dices else 0.0
        mean_base_iou = float(np.mean(baseline_ious)) if baseline_ious else 0.0
        mean_rb_dice = float(np.mean(rb_dices)) if rb_dices else 0.0
        mean_rb_iou = float(np.mean(rb_ious)) if rb_ious else 0.0

        return {
            "condition": condition_name,
            "baseline_dice": mean_base_dice,
            "baseline_iou": mean_base_iou,
            "rb_unet_dice": mean_rb_dice,
            "rb_unet_iou": mean_rb_iou,
            "delta_dice": mean_rb_dice - mean_base_dice,
            "delta_iou": mean_rb_iou - mean_base_iou,
        }

    def run_benchmark(self, dataloader: DataLoader) -> List[Dict[str, Any]]:
        results = []
        print("\n" + "=" * 70)
        print("ROBUSTNESS BENCHMARK: STANDARD U-NET vs RB-UNET")
        print("=" * 70)
        print(f"{'Condition':<18} | {'U-Net Dice':<10} | {'RB-UNet Dice':<12} | {'Delta Dice':<10}")
        print("-" * 70)

        for name, params in self.CONDITIONS:
            res = self.evaluate_condition(dataloader, name, params)
            results.append(res)
            print(
                f"{res['condition']:<18} | "
                f"{res['baseline_dice']:<10.4f} | "
                f"{res['rb_unet_dice']:<12.4f} | "
                f"{res['delta_dice']:>+10.4f}"
            )

        print("=" * 70)
        return results
