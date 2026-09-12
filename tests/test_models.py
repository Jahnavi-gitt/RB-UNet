"""Unit tests for model architecture forward passes and tensor shapes."""

import pytest
import torch

from src.models.unet import UNet
from src.models.rb_unet import RBUNet


def test_standard_unet_forward():
    model = UNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128, 256])
    x = torch.randn(2, 3, 64, 64)
    out = model(x)

    assert out.shape == (2, 1, 64, 64)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_standard_unet_gradient_flow():
    model = UNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128, 256])
    x = torch.randn(2, 3, 64, 64)
    target = torch.ones(2, 1, 64, 64)

    out = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, target)
    loss.backward()

    # Verify gradients exist on encoder and decoder weights
    assert model.enc1.double_conv[0].weight.grad is not None
    assert model.out_conv.weight.grad is not None


def test_rb_unet_forward():
    model = RBUNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128, 256], boundary_head_channels=16)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)

    assert isinstance(out, dict)
    assert "seg_logits" in out
    assert "boundary_logits" in out
    assert out["seg_logits"].shape == (2, 1, 64, 64)
    assert out["boundary_logits"].shape == (2, 1, 64, 64)
    assert not torch.isnan(out["seg_logits"]).any()
    assert not torch.isnan(out["boundary_logits"]).any()


def test_rb_unet_dual_gradient_flow():
    model = RBUNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128, 256], boundary_head_channels=16)
    x = torch.randn(2, 3, 64, 64)
    target_seg = torch.ones(2, 1, 64, 64)
    target_bnd = torch.zeros(2, 1, 64, 64)

    out = model(x)
    loss_seg = torch.nn.functional.binary_cross_entropy_with_logits(out["seg_logits"], target_seg)
    loss_bnd = torch.nn.functional.binary_cross_entropy_with_logits(out["boundary_logits"], target_bnd)
    total_loss = loss_seg + 0.5 * loss_bnd
    total_loss.backward()

    # Both heads and backbone must have valid gradients
    assert model.seg_head.weight.grad is not None
    assert model.boundary_head.head[0].weight.grad is not None
    assert model.enc1.double_conv[0].weight.grad is not None
