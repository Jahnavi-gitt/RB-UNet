"""Evaluation metrics for lesion segmentation: Dice, IoU, and Boundary F1."""

from typing import Dict, Union
import numpy as np
import torch


def compute_dice(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Compute Dice Similarity Coefficient (DSC) between prediction and target.
    
    DSC = (2 * |X ∩ Y|) / (|X| + |Y|)
    Handles empty masks cleanly:
    - If both pred and target are empty: 1.0
    - If only one is empty: 0.0
    """
    if isinstance(pred, torch.Tensor):
        if pred.dtype in (torch.float32, torch.float64) and (pred.min() < 0 or pred.max() > 1):
            pred = torch.sigmoid(pred)
        pred_bin = (pred > threshold).float()
        target_bin = (target > threshold).float()

        intersection = torch.sum(pred_bin * target_bin).item()
        pred_sum = torch.sum(pred_bin).item()
        target_sum = torch.sum(target_bin).item()
    else:
        pred_arr = np.array(pred)
        target_arr = np.array(target)
        if pred_arr.dtype in (np.float32, np.float64) and (pred_arr.min() < 0 or pred_arr.max() > 1):
            pred_arr = 1.0 / (1.0 + np.exp(-pred_arr))
        pred_bin = (pred_arr > threshold).astype(np.float32)
        target_bin = (target_arr > threshold).astype(np.float32)

        intersection = float(np.sum(pred_bin * target_bin))
        pred_sum = float(np.sum(pred_bin))
        target_sum = float(np.sum(target_bin))

    total = pred_sum + target_sum
    if total == 0:
        # Both masks are completely empty
        return 1.0

    dice = (2.0 * intersection) / (total + eps)
    return float(dice)


def compute_iou(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Compute Intersection over Union (IoU / Jaccard Index).
    
    IoU = |X ∩ Y| / |X ∪ Y|
    """
    if isinstance(pred, torch.Tensor):
        if pred.dtype in (torch.float32, torch.float64) and (pred.min() < 0 or pred.max() > 1):
            pred = torch.sigmoid(pred)
        pred_bin = (pred > threshold).float()
        target_bin = (target > threshold).float()

        intersection = torch.sum(pred_bin * target_bin).item()
        union = torch.sum((pred_bin + target_bin) > 0).item()
    else:
        pred_arr = np.array(pred)
        target_arr = np.array(target)
        if pred_arr.dtype in (np.float32, np.float64) and (pred_arr.min() < 0 or pred_arr.max() > 1):
            pred_arr = 1.0 / (1.0 + np.exp(-pred_arr))
        pred_bin = (pred_arr > threshold).astype(np.float32)
        target_bin = (target_arr > threshold).astype(np.float32)

        intersection = float(np.sum(pred_bin * target_bin))
        union = float(np.sum((pred_bin + target_bin) > 0))

    if union == 0:
        return 1.0

    iou = intersection / (union + eps)
    return float(iou)


def evaluate_batch_metrics(
    pred_logits: torch.Tensor,
    target_masks: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute average Dice and IoU across a batch."""
    batch_size = pred_logits.shape[0]
    probs = torch.sigmoid(pred_logits)

    dice_scores = []
    iou_scores = []

    for b in range(batch_size):
        d = compute_dice(probs[b], target_masks[b], threshold=threshold)
        i = compute_iou(probs[b], target_masks[b], threshold=threshold)
        dice_scores.append(d)
        iou_scores.append(i)

    return {
        "dice": float(np.mean(dice_scores)),
        "iou": float(np.mean(iou_scores)),
    }
