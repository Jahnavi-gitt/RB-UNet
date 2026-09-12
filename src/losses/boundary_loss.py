"""Boundary supervision loss for auxiliary lesion contour prediction."""

import torch
import torch.nn as nn
from .dice_loss import SoftDiceLoss


class BoundaryLoss(nn.Module):
    """Auxiliary boundary supervision loss.
    
    Combines Weighted BCE (to handle extreme boundary sparsity) and Soft Dice loss.
    """

    def __init__(self, pos_weight: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.dice_loss = SoftDiceLoss(eps=eps)

    def forward(self, boundary_logits: torch.Tensor, boundary_targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(boundary_logits, boundary_targets)
        dice = self.dice_loss(boundary_logits, boundary_targets)
        return bce + dice
