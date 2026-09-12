"""Unit tests for loss stability, gradient computation, and edge cases."""

import pytest
import torch

from src.losses.dice_loss import SoftDiceLoss
from src.losses.segmentation_loss import CombinedSegmentationLoss
from src.losses.boundary_loss import BoundaryLoss
from src.losses.consistency_loss import ConsistencyLoss


def test_dice_loss_finite_and_gradients():
    loss_fn = SoftDiceLoss()
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = (torch.rand(2, 1, 32, 32) > 0.5).float()

    loss = loss_fn(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_combined_segmentation_loss():
    loss_fn = CombinedSegmentationLoss(dice_weight=1.0, bce_weight=1.0)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.ones(2, 1, 32, 32)

    loss = loss_fn(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_boundary_loss():
    loss_fn = BoundaryLoss()
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = (torch.rand(2, 1, 32, 32) > 0.8).float()

    loss = loss_fn(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_consistency_loss():
    loss_fn = ConsistencyLoss()
    clean_logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    corrupt_logits = clean_logits + 0.1 * torch.randn(2, 1, 32, 32)

    loss = loss_fn(clean_logits, corrupt_logits)
    loss.backward()

    assert torch.isfinite(loss)
    assert clean_logits.grad is not None
