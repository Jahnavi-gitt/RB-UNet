"""Consistency loss between clean and corrupted model predictions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """Mean Squared Error consistency loss between predicted probability maps."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, clean_logits: torch.Tensor, corrupt_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clean_logits: Predictions from clean image [B, 1, H, W]
            corrupt_logits: Predictions from corrupted image [B, 1, H, W]
        Returns:
            Scalar consistency loss
        """
        prob_clean = torch.sigmoid(clean_logits)
        prob_corrupt = torch.sigmoid(corrupt_logits)
        return self.mse(prob_clean, prob_corrupt)
