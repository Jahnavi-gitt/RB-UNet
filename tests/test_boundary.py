"""Unit tests for boundary extraction and boundary head."""

import pytest
import torch

from src.models.boundary_head import BoundaryHead, extract_boundary_targets


def test_boundary_target_extraction():
    # Create a 20x20 binary mask with a 10x10 centered square
    mask = torch.zeros(1, 1, 20, 20)
    mask[0, 0, 5:15, 5:15] = 1.0

    boundary = extract_boundary_targets(mask, kernel_size=3)

    assert boundary.shape == mask.shape
    # Values must be strictly in {0.0, 1.0}
    unique_vals = torch.unique(boundary).tolist()
    for v in unique_vals:
        assert v in (0.0, 1.0)

    # Boundary sum should be non-zero and less than the total area
    assert boundary.sum() > 0
    assert boundary.sum() < mask.sum()


def test_boundary_empty_mask():
    mask = torch.zeros(1, 1, 20, 20)
    boundary = extract_boundary_targets(mask, kernel_size=3)
    assert boundary.sum() == 0.0


def test_boundary_head_forward():
    head = BoundaryHead(in_channels=64, hidden_channels=32)
    feat = torch.randn(2, 64, 32, 32)
    out = head(feat)
    assert out.shape == (2, 1, 32, 32)
    assert not torch.isnan(out).any()
