"""Evaluation engine for calculating dataset-level metrics and exporting predictions."""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_dice, compute_iou


class Evaluator:
    """Evaluates segmentation models across dataset splits."""

    def __init__(self, model: nn.Module, device: torch.device, threshold: float = 0.5):
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, desc: str = "Evaluating") -> Dict[str, Any]:
        self.model.eval()
        dice_scores = []
        iou_scores = []
        sample_results = []

        for batch in tqdm(dataloader, desc=desc, leave=False):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            sample_ids = batch.get("id", ["unknown"] * len(images))

            out = self.model(images)
            if isinstance(out, dict):
                seg_logits = out["seg_logits"]
                boundary_logits = out.get("boundary_logits")
            else:
                seg_logits = out
                boundary_logits = None

            probs = torch.sigmoid(seg_logits)

            for b in range(len(images)):
                d = compute_dice(probs[b], masks[b], threshold=self.threshold)
                i = compute_iou(probs[b], masks[b], threshold=self.threshold)
                dice_scores.append(d)
                iou_scores.append(i)

                sample_results.append({
                    "id": sample_ids[b] if isinstance(sample_ids, list) else sample_ids[b],
                    "dice": d,
                    "iou": i,
                })

        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

        return {
            "mean_dice": mean_dice,
            "mean_iou": mean_iou,
            "sample_count": len(dice_scores),
            "samples": sample_results,
        }
