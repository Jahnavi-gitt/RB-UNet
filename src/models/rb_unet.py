"""RB-UNet: Robust Boundary-Consistent U-Net for Medical Lesion Segmentation."""

from typing import Dict, List
import torch
import torch.nn as nn
from .boundary_head import BoundaryHead
from .unet import DoubleConv


class RBUNet(nn.Module):
    """Robust Boundary-Consistent U-Net.
    
    Shares U-Net encoder-decoder backbone, branching at the final feature representation
    into:
    1. Primary lesion segmentation head
    2. Lightweight auxiliary lesion boundary prediction head
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512, 1024],
        boundary_head_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(features[0], features[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(features[1], features[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(features[2], features[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[4])

        # Decoder (Expanding Path)
        self.up4 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features[4], features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features[3], features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features[2], features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features[1], features[0])

        # Primary Segmentation Head
        self.seg_head = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Auxiliary Boundary Head
        self.boundary_head = BoundaryHead(
            in_channels=features[0],
            hidden_channels=boundary_head_channels,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            Dict containing:
                - 'seg_logits': [B, 1, H, W]
                - 'boundary_logits': [B, 1, H, W]
        """
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        # Bottleneck
        b = self.bottleneck(self.pool4(x4))

        # Decoder
        d4 = self.dec4(torch.cat([x4, self.up4(b)], dim=1))
        d3 = self.dec3(torch.cat([x3, self.up3(d4)], dim=1))
        d2 = self.dec2(torch.cat([x2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([x1, self.up1(d2)], dim=1))

        # Dual Heads
        seg_logits = self.seg_head(d1)
        boundary_logits = self.boundary_head(d1)

        return {
            "seg_logits": seg_logits,
            "boundary_logits": boundary_logits,
        }
