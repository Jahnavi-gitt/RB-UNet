"""Deterministic and training-time image corruptions for robustness evaluation."""

import random
from typing import Dict, Tuple
import torch
import torchvision.transforms.functional as TF


def apply_gaussian_noise(image: torch.Tensor, std: float = 0.08, seed: int = 42) -> torch.Tensor:
    """Add zero-mean Gaussian noise to tensor [B, C, H, W] with fixed reproducible seed."""
    gen = torch.Generator(device=image.device).manual_seed(seed)
    noise = torch.randn(image.shape, generator=gen, device=image.device, dtype=image.dtype) * std
    return image + noise


def apply_gaussian_blur(image: torch.Tensor, kernel_size: int = 5, sigma: float = 1.5) -> torch.Tensor:
    """Apply 2D Gaussian blur."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def apply_contrast_shift(image: torch.Tensor, factor: float = 0.6) -> torch.Tensor:
    """Scale contrast: (image - mean) * factor + mean."""
    mean = torch.mean(image, dim=(-2, -1), keepdim=True)
    return (image - mean) * factor + mean


def apply_brightness_shift(image: torch.Tensor, offset: float = -0.2) -> torch.Tensor:
    """Shift brightness by constant additive offset."""
    return image + offset


def apply_speckle_noise(image: torch.Tensor, std: float = 0.08) -> torch.Tensor:
    """Add multiplicative speckle noise: I + I * N(0, std^2)."""
    noise = torch.randn_like(image) * std
    return image + image * noise


def apply_named_corruption(
    image: torch.Tensor,
    name: str,
    severity_params: Dict = None,
) -> torch.Tensor:
    """Apply named corruption deterministically for robustness testing."""
    params = severity_params or {}
    name = name.lower()

    if "noise" in name:
        std = params.get("std", 0.08)
        return apply_gaussian_noise(image, std=std)
    elif "blur" in name:
        kernel_size = params.get("kernel_size", 5)
        sigma = params.get("sigma", 1.5)
        return apply_gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
    elif "contrast" in name:
        factor = params.get("factor", 0.6)
        return apply_contrast_shift(image, factor=factor)
    elif "brightness" in name:
        offset = params.get("offset", -0.2)
        return apply_brightness_shift(image, offset=offset)
    elif "speckle" in name:
        std = params.get("std", 0.08)
        return apply_speckle_noise(image, std=std)
    elif name == "clean":
        return image
    else:
        raise ValueError(f"Unknown corruption name: {name}")


def apply_random_corruption(image: torch.Tensor, config: Dict = None) -> Tuple[torch.Tensor, str]:
    """Randomly apply one enabled corruption during consistency training."""
    corruptions = [
        ("gaussian_noise", lambda x: apply_gaussian_noise(x, std=0.08)),
        ("gaussian_blur", lambda x: apply_gaussian_blur(x, kernel_size=5, sigma=1.5)),
        ("contrast", lambda x: apply_contrast_shift(x, factor=0.6)),
        ("brightness", lambda x: apply_brightness_shift(x, offset=-0.2)),
    ]
    name, func = random.choice(corruptions)
    return func(image), name
