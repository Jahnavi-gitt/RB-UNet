"""Unit tests for dataset loading, discovery, and synchronized transforms."""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from src.datasets.isic_dataset import ISICDataset
from src.datasets.split_discovery import discover_dataset_splits
from src.datasets.transforms import get_transforms


@pytest.fixture
def mock_dataset_dir(tmp_path):
    """Creates a temporary mock ISIC folder structure."""
    train_img_dir = tmp_path / "ISIC2018_Task1-1_Training_Input"
    train_mask_dir = tmp_path / "ISIC2018_Task1_Training_GroundTruth"
    train_img_dir.mkdir(parents=True)
    train_mask_dir.mkdir(parents=True)

    for i in range(3):
        sid = f"ISIC_{i:07d}"
        img_arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        mask_arr = ((np.random.rand(64, 64) > 0.5) * 255).astype(np.uint8)

        Image.fromarray(img_arr).save(train_img_dir / f"{sid}.jpg")
        Image.fromarray(mask_arr).save(train_mask_dir / f"{sid}_segmentation.png")

    return tmp_path


def test_dataset_discovery(mock_dataset_dir):
    result = discover_dataset_splits(str(mock_dataset_dir))
    assert result["exists"] is True
    assert len(result["train"]) == 3
    assert result["train"][0]["id"] == "ISIC_0000000"
    assert Path(result["train"][0]["image_path"]).exists()
    assert Path(result["train"][0]["mask_path"]).exists()


def test_isic_dataset_loading(mock_dataset_dir):
    result = discover_dataset_splits(str(mock_dataset_dir))
    transform = get_transforms(image_size=128, is_training=False)
    dataset = ISICDataset(result["train"], transform=transform)

    assert len(dataset) == 3
    item = dataset[0]

    assert "image" in item
    assert "mask" in item
    assert "id" in item

    # Check tensor shapes
    assert item["image"].shape == (3, 128, 128)
    assert item["mask"].shape == (1, 128, 128)

    # Check mask is strictly binary {0.0, 1.0}
    unique_vals = torch.unique(item["mask"]).tolist()
    for v in unique_vals:
        assert v in (0.0, 1.0)


def test_mask_nearest_neighbor_preservation():
    """Verify that mask values do not become non-binary after resizing."""
    transform = get_transforms(image_size=128, is_training=False)
    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    # Binary mask
    mask_arr = np.zeros((64, 64), dtype=np.uint8)
    mask_arr[20:45, 20:45] = 255
    mask = Image.fromarray(mask_arr)

    img_tensor, mask_tensor = transform(img, mask)
    unique_vals = torch.unique(mask_tensor).tolist()
    assert all(v in (0.0, 1.0) for v in unique_vals)
