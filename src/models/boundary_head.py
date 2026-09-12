"""Auxiliary boundary prediction head and morphological target extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_boundary_targets(masks: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Extract binary lesion boundary targets using morphological dilation - erosion.
    
    Args:
        masks: Binary ground truth mask tensor [B, 1, H, W] with values in {0.0, 1.0}.
        kernel_size: Morphological structuring element size (default: 3).
        
    Returns:
        boundary_target: Binary boundary tensor [B, 1, H, W] with values in {0.0, 1.0}.
    """
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)

    pad = kernel_size // 2

    # Morphological Dilation using MaxPool2d
    dilation = F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=pad)

    # Morphological Erosion using Duality: 1 - Dilation(1 - Mask)
    inverted = 1.0 - masks
    erosion = 1.0 - F.max_pool2d(inverted, kernel_size=kernel_size, stride=1, padding=pad)

    # Boundary is Dilation - Erosion
    boundary = dilation - erosion

    # Clamp to strictly binary {0.0, 1.0}
    boundary = torch.clamp(boundary, 0.0, 1.0)
    return boundary


class BoundaryHead(nn.Module):
    """Lightweight auxiliary head predicting lesion boundary logits."""

    def __init__(self, in_channels: int = 64, hidden_channels: int = 32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map from U-Net decoder, shape [B, in_channels, H, W]
        Returns:
            Boundary logits, shape [B, 1, H, W]
        """
        return self.head(x)
