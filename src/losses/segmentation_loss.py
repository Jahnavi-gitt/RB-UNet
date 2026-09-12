"""Combined BCE + Soft Dice loss for primary segmentation head."""

import torch
import torch.nn as nn
from .dice_loss import SoftDiceLoss


class CombinedSegmentationLoss(nn.Module):
    """Combines Binary Cross Entropy with Soft Dice Loss."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = SoftDiceLoss(eps=eps)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice
