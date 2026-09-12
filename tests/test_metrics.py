"""Unit tests for mathematical correctness and edge cases of Dice and IoU."""

import numpy as np
import pytest
import torch

from src.evaluation.metrics import compute_dice, compute_iou, evaluate_batch_metrics


def test_perfect_prediction():
    pred = torch.ones((1, 64, 64))
    target = torch.ones((1, 64, 64))
    assert compute_dice(pred, target) == pytest.approx(1.0, rel=1e-5)
    assert compute_iou(pred, target) == pytest.approx(1.0, rel=1e-5)


def test_zero_overlap():
    pred = torch.zeros((1, 64, 64))
    pred[:, :32, :] = 1.0
    target = torch.zeros((1, 64, 64))
    target[:, 32:, :] = 1.0
    assert compute_dice(pred, target) == pytest.approx(0.0, abs=1e-5)
    assert compute_iou(pred, target) == pytest.approx(0.0, abs=1e-5)


def test_partial_overlap():
    # 50% overlap calculation
    pred = torch.zeros((1, 10, 10))
    target = torch.zeros((1, 10, 10))
    pred[:, 0:4, :] = 1.0    # 40 pixels
    target[:, 2:6, :] = 1.0  # 40 pixels
    # Intersection = rows 2,3 (20 pixels)
    # Total sum = 40 + 40 = 80 -> Dice = (2 * 20) / 80 = 0.5
    # Union = rows 0,1,2,3,4,5 (60 pixels) -> IoU = 20 / 60 = 0.33333
    assert compute_dice(pred, target) == pytest.approx(0.5, rel=1e-4)
    assert compute_iou(pred, target) == pytest.approx(1.0 / 3.0, rel=1e-4)


def test_both_empty_masks():
    pred = torch.zeros((1, 64, 64))
    target = torch.zeros((1, 64, 64))
    assert compute_dice(pred, target) == 1.0
    assert compute_iou(pred, target) == 1.0


def test_one_empty_mask():
    pred = torch.zeros((1, 64, 64))
    target = torch.ones((1, 64, 64))
    assert compute_dice(pred, target) == pytest.approx(0.0, abs=1e-5)
    assert compute_iou(pred, target) == pytest.approx(0.0, abs=1e-5)


def test_batch_evaluation():
    logits = torch.randn(4, 1, 32, 32)
    targets = (torch.rand(4, 1, 32, 32) > 0.5).float()
    metrics = evaluate_batch_metrics(logits, targets)
    assert "dice" in metrics
    assert "iou" in metrics
    assert 0.0 <= metrics["dice"] <= 1.0
    assert 0.0 <= metrics["iou"] <= 1.0
