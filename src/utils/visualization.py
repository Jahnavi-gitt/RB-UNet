"""High-resolution visualization tools for lesion segmentation and comparison."""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize_image(
    image_tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """Denormalize tensor [3, H, W] to uint8 RGB numpy array."""
    img = image_tensor.detach().cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = torch.clamp(img, 0.0, 1.0)
    img_np = img.permute(1, 2, 0).numpy()
    return (img_np * 255).astype(np.uint8)


def create_overlay(image_np: np.ndarray, mask_np: np.ndarray, color=(0, 210, 180), alpha=0.4) -> np.ndarray:
    """Create semi-transparent lesion overlay on original RGB image."""
    overlay = image_np.copy().astype(np.float32)
    binary_mask = (mask_np > 0.5)

    for c in range(3):
        overlay[:, :, c] = np.where(
            binary_mask,
            overlay[:, :, c] * (1.0 - alpha) + color[c] * alpha,
            overlay[:, :, c],
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_comparative_panel(
    image: np.ndarray,
    ground_truth: Optional[np.ndarray],
    baseline_pred: Optional[np.ndarray],
    rb_unet_pred: Optional[np.ndarray],
    boundary_pred: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Lesion Segmentation Comparison",
) -> None:
    """Generate high-resolution comparative panel."""
    panels = [("Original", image)]
    if ground_truth is not None:
        panels.append(("Ground Truth", ground_truth))
    if baseline_pred is not None:
        panels.append(("Standard U-Net", baseline_pred))
    if rb_unet_pred is not None:
        panels.append(("RB-UNet (Proposed)", rb_unet_pred))
    if boundary_pred is not None:
        panels.append(("Predicted Boundary", boundary_pred))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, img_data) in zip(axes, panels):
        if img_data.ndim == 2:
            ax.imshow(img_data, cmap="gray", vmin=0, vmax=1 if img_data.max() <= 1.0 else 255)
        else:
            ax.imshow(img_data)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curves(history: List[dict], save_path: str) -> None:
    """Plot convergence curves for train loss and validation Dice/IoU."""
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    dices = [h["val_dice"] for h in history]
    ious = [h["val_iou"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(epochs, losses, "o-", color="#E53E3E", label="Train Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Convergence")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Validation Dice & IoU
    ax2.plot(epochs, dices, "o-", color="#00D2B4", label="Val Dice", linewidth=2)
    ax2.plot(epochs, ious, "s-", color="#3182CE", label="Val IoU", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metric Progression")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=200, bbox_inches="tight")
    plt.close()
