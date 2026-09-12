"""Synchronized image and mask transformations for ISIC lesion segmentation."""

import random
from typing import Callable, Dict, Tuple
from PIL import Image
import torch
import torchvision.transforms.functional as TF


class SynchronizedTransform:
    """Applies synchronized spatial transforms to both image and mask."""

    def __init__(
        self,
        image_size: int = 256,
        is_training: bool = False,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.image_size = (image_size, image_size)
        self.is_training = is_training
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize image with bilinear interpolation
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        # Resize mask strictly with NEAREST interpolation to preserve binary integer values
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        # Synchronized random spatial augmentations during training only
        if self.is_training:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random mild rotation (-15 to +15 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-15.0, 15.0)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Convert image to tensor [3, H, W] in range [0.0, 1.0]
        image_tensor = TF.to_tensor(image)
        # Normalize image
        image_tensor = TF.normalize(image_tensor, mean=self.normalize_mean, std=self.normalize_std)

        # Convert mask to single-channel binary float tensor [1, H, W] with values in {0.0, 1.0}
        mask_tensor = TF.to_tensor(mask)
        if mask_tensor.shape[0] > 1:
            mask_tensor = mask_tensor[0:1, :, :]  # Take first channel if RGB mask
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor


def get_transforms(
    image_size: int = 256,
    is_training: bool = False,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> SynchronizedTransform:
    """Factory function for synchronized transforms."""
    return SynchronizedTransform(
        image_size=image_size,
        is_training=is_training,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
