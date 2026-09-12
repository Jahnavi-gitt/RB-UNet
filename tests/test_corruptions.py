"""Unit tests for deterministic image corruptions."""

import pytest
import torch

from src.corruption.corruptions import (
    apply_gaussian_noise,
    apply_gaussian_blur,
    apply_contrast_shift,
    apply_brightness_shift,
    apply_named_corruption,
)


def test_corruptions_preserve_shapes():
    x = torch.rand(2, 3, 64, 64)

    for name in ["gaussian_noise", "gaussian_blur", "contrast", "brightness", "clean"]:
        corrupted = apply_named_corruption(x, name)
        assert corrupted.shape == x.shape
        assert corrupted.dtype == x.dtype
        assert not torch.isnan(corrupted).any()


def test_gaussian_noise_modifies_values():
    x = torch.zeros(2, 3, 32, 32)
    noisy = apply_gaussian_noise(x, std=0.1)
    assert not torch.equal(x, noisy)
    assert abs(noisy.mean().item()) < 0.05
