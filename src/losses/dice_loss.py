"""Soft Dice Loss for end-to-end differentiable optimization."""

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """Numerically stable Soft Dice Loss."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized network predictions [B, 1, H, W]
            targets: Binary ground truth targets [B, 1, H, W] in {0.0, 1.0}
        Returns:
            Scalar soft dice loss
        """
        probs = torch.sigmoid(logits)

        # Flatten across spatial dimensions
        probs_flat = probs.view(probs.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        cardinality = torch.sum(probs_flat + targets_flat, dim=1)

        dice_coeff = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        loss = 1.0 - dice_coeff
        return loss.mean()
