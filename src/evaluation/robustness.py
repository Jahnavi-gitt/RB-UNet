"""Deterministic robustness benchmark across image degradation conditions."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

    SLUG_MAP = {
        "Clean": "clean",
        "Gaussian Noise": "gaussian_noise",
        "Gaussian Blur": "gaussian_blur",
        "Contrast Shift": "contrast",
        "Brightness Shift": "brightness",
    }

    def __init__(
        self,
        baseline_model: nn.Module,
        rb_unet_model: nn.Module,
        device: torch.device,
        threshold: float = 0.5,
        save_dir: Optional[str] = "outputs/robustness",
    ):
        self.baseline = baseline_model.to(device).eval()
        self.rb_unet = rb_unet_model.to(device).eval()
        self.device = device
        self.threshold = threshold
        self.save_dir = Path(save_dir) if save_dir else None

    @torch.no_grad()
    def evaluate_condition(
        self,
        dataloader: DataLoader,
        condition_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        baseline_dices = []
        baseline_ious = []
        rb_dices = []
        rb_ious = []

        slug = self.SLUG_MAP.get(condition_name, condition_name.lower().replace(" ", "_"))
        visual_saved = False
        saved_paths = {}

        # Precompute mean and std tensors for denormalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        for batch_idx, batch in enumerate(dataloader):
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

            # Save qualitative visual artifacts for the first batch
            if self.save_dir and not visual_saved and len(images) > 0:
                cond_dir = self.save_dir / slug
                cond_dir.mkdir(parents=True, exist_ok=True)

                # Denormalize corrupted image to [0, 255] RGB
                denorm_img = (corrupt_images[0:1] * std + mean).clamp(0.0, 1.0)
                denorm_np = (denorm_img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                # Predictions
                base_mask_np = ((prob_base[0, 0] > self.threshold).cpu().numpy() * 255).astype(np.uint8)
                rb_mask_np = ((prob_rb[0, 0] > self.threshold).cpu().numpy() * 255).astype(np.uint8)
                gt_mask_np = (masks[0, 0].cpu().numpy() * 255).astype(np.uint8)

                input_file = cond_dir / "input.png"
                base_pred_file = cond_dir / "baseline_prediction.png"
                rb_pred_file = cond_dir / "rb_unet_prediction.png"
                gt_file = cond_dir / "ground_truth.png"

                Image.fromarray(denorm_np).save(input_file)
                Image.fromarray(base_mask_np).save(base_pred_file)
                Image.fromarray(rb_mask_np).save(rb_pred_file)
                Image.fromarray(gt_mask_np).save(gt_file)

                saved_paths = {
                    "input_url": f"/outputs/robustness/{slug}/input.png",
                    "baseline_prediction_url": f"/outputs/robustness/{slug}/baseline_prediction.png",
                    "rb_unet_prediction_url": f"/outputs/robustness/{slug}/rb_unet_prediction.png",
                    "ground_truth_url": f"/outputs/robustness/{slug}/ground_truth.png",
                }
                visual_saved = True

            for b in range(len(images)):
                baseline_dices.append(compute_dice(prob_base[b], masks[b], threshold=self.threshold))
                baseline_ious.append(compute_iou(prob_base[b], masks[b], threshold=self.threshold))
                rb_dices.append(compute_dice(prob_rb[b], masks[b], threshold=self.threshold))
                rb_ious.append(compute_iou(prob_rb[b], masks[b], threshold=self.threshold))

        mean_base_dice = float(np.mean(baseline_dices)) if baseline_dices else 0.0
        mean_base_iou = float(np.mean(baseline_ious)) if baseline_ious else 0.0
        mean_rb_dice = float(np.mean(rb_dices)) if rb_dices else 0.0
        mean_rb_iou = float(np.mean(rb_ious)) if rb_ious else 0.0

        res_dict = {
            "condition": condition_name,
            "slug": slug,
            "baseline_dice": round(mean_base_dice, 4),
            "baseline_iou": round(mean_base_iou, 4),
            "rb_unet_dice": round(mean_rb_dice, 4),
            "rb_unet_iou": round(mean_rb_iou, 4),
            "delta_dice": round(mean_rb_dice - mean_base_dice, 4),
            "delta_iou": round(mean_rb_iou - mean_base_iou, 4),
        }
        res_dict.update(saved_paths)
        return res_dict

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
