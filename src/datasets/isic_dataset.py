"""PyTorch Dataset implementation for ISIC lesion boundary segmentation."""

from pathlib import Path
from typing import Callable, Dict, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    """ISIC 2018 Lesion Boundary Segmentation Dataset.
    
    Reads image and ground-truth mask pairs, applies synchronized transformations,
    and returns formatted tensors suitable for model training and evaluation.
    """

    def __init__(
        self,
        samples: List[Dict[str, str]],
        transform: Optional[Callable] = None,
        return_boundary: bool = False,
        boundary_transform: Optional[Callable] = None,
    ):
        """
        Args:
            samples: List of dictionaries with keys 'id', 'image_path', 'mask_path'.
            transform: Callable transform accepting (image, mask) and returning (image_tensor, mask_tensor).
            return_boundary: Whether to compute and return ground truth boundary map.
            boundary_transform: Optional callable to generate boundary from mask tensor.
        """
        self.samples = samples
        self.transform = transform
        self.return_boundary = return_boundary
        self.boundary_transform = boundary_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_meta = self.samples[idx]
        sample_id = sample_meta["id"]
        image_path = sample_meta["image_path"]
        mask_path = sample_meta["mask_path"]

        # Load RGB image
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # Load binary mask
        with Image.open(mask_path) as msk:
            mask = msk.convert("L")

        # Apply synchronized transforms
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask)
        else:
            from torchvision.transforms.functional import to_tensor
            image_tensor = to_tensor(image)
            mask_tensor = (to_tensor(mask) > 0.5).float()

        item = {
            "image": image_tensor,
            "mask": mask_tensor,
            "id": sample_id,
        }

        # Optionally add boundary target
        if self.return_boundary and self.boundary_transform is not None:
            # boundary_transform takes [1, H, W] mask tensor and returns [1, H, W] boundary tensor
            boundary_tensor = self.boundary_transform(mask_tensor.unsqueeze(0)).squeeze(0)
            item["boundary"] = boundary_tensor

        return item
