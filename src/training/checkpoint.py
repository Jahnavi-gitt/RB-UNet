"""Metadata-rich model checkpointing and weight management."""

from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    filepath: str,
    epoch: int,
    val_dice: float,
    val_iou: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
) -> None:
    """Save model checkpoint with full reproducibility metadata."""
    save_path = Path(filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "val_dice": float(val_dice),
        "val_iou": float(val_iou),
        "config": config or {},
        "is_best": is_best,
    }

    torch.save(checkpoint_data, str(save_path))
    if is_best:
        best_path = save_path.parent / "best_model.pth"
        torch.save(checkpoint_data, str(best_path))
        print(f"[Checkpoint] Saved new BEST model to {best_path} (Val Dice: {val_dice:.4f}, IoU: {val_iou:.4f})")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load model weights and metadata from checkpoint."""
    checkpoint_path = Path(filepath)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"[Checkpoint] Loaded weights from {filepath} (Epoch {checkpoint.get('epoch')}, Val Dice: {checkpoint.get('val_dice'):.4f})")
    return checkpoint


def build_model_from_checkpoint(filepath: str, device: str = "cpu") -> nn.Module:
    """Dynamically reconstruct the exact model architecture directly from saved state dict."""
    checkpoint_path = Path(filepath)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]

    f0 = state["enc1.double_conv.0.weight"].shape[0]
    f1 = state["enc2.double_conv.0.weight"].shape[0]
    f2 = state["enc3.double_conv.0.weight"].shape[0]
    f3 = state["enc4.double_conv.0.weight"].shape[0]
    f4 = state["bottleneck.double_conv.0.weight"].shape[0]
    features = [f0, f1, f2, f3, f4]

    from src.models.unet import UNet
    from src.models.rb_unet import RBUNet

    is_rb = "boundary_head.head.0.weight" in state
    if is_rb:
        bnd_channels = state["boundary_head.head.0.weight"].shape[0]
        model = RBUNet(in_channels=3, out_channels=1, features=features, boundary_head_channels=bnd_channels)
    else:
        model = UNet(in_channels=3, out_channels=1, features=features)

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
