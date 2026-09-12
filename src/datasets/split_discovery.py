"""Automatic dataset discovery and split validator for ISIC 2018.

Inspects the configured dataset root directory, dynamically identifies
train, validation, and test splits, and securely pairs images with their
corresponding binary ground-truth masks.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _extract_isic_id(filename: str) -> Optional[str]:
    """Extract ISIC ID from filename (e.g. 'ISIC_0000001' from 'ISIC_0000001.jpg')."""
    match = re.search(r"(ISIC_\d+)", filename, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _find_image_files(directory: Path) -> List[Path]:
    """Find all image files in a directory."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not directory.exists():
        return []
    return [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]


def _build_id_map(files: List[Path]) -> Dict[str, Path]:
    """Map ISIC ID -> file path."""
    id_map = {}
    for f in files:
        sid = _extract_isic_id(f.name)
        if sid:
            id_map[sid] = f
    return id_map


def _match_pairs(
    image_dir: Optional[Path],
    mask_dir: Optional[Path]
) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
    """Pair images with their masks based on ISIC IDs."""
    pairs = []
    orphaned_images = []
    orphaned_masks = []

    if not image_dir or not image_dir.exists():
        return pairs, orphaned_images, orphaned_masks

    image_files = _find_image_files(image_dir)
    image_map = _build_id_map(image_files)

    if mask_dir and mask_dir.exists():
        mask_files = _find_image_files(mask_dir)
        mask_map = _build_id_map(mask_files)
    else:
        mask_map = {}

    for sid, img_path in image_map.items():
        if sid in mask_map:
            pairs.append({
                "id": sid,
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_map[sid].resolve()),
            })
        else:
            orphaned_images.append(sid)

    for sid in mask_map:
        if sid not in image_map:
            orphaned_masks.append(sid)

    pairs.sort(key=lambda x: x["id"])
    return pairs, orphaned_images, orphaned_masks


def discover_dataset_splits(dataset_root: str) -> Dict:
    """Dynamically scan dataset_root and discover train, val, and test splits.
    
    Supports official ISIC 2018 folder conventions:
    - ISIC2018_Task1-1_Training_Input / ISIC2018_Task1_Training_GroundTruth
    - ISIC2018_Task1-2_Validation_Input / ISIC2018_Task1_Validation_GroundTruth
    - ISIC2018_Task1-3_Test_Input / ISIC2018_Task1_Test_GroundTruth
    As well as generic 'train', 'val' / 'validation', 'test' directory patterns.
    """
    root_path = Path(dataset_root).resolve()
    if not root_path.exists():
        return {
            "root": str(root_path),
            "exists": False,
            "train": [],
            "val": [],
            "test": [],
            "stats": {"total_train": 0, "total_val": 0, "total_test": 0, "error": f"Path '{root_path}' does not exist."},
        }

    # Find candidate directories
    all_subdirs = [d for d in root_path.rglob("*") if d.is_dir()] + [root_path]

    def _find_dir_by_pattern(patterns: List[str]) -> Optional[Path]:
        # Iterate over patterns first so specific patterns have strict priority over general fallbacks
        for pat in patterns:
            pat_lower = pat.lower()
            for d in all_subdirs:
                if pat_lower == d.name.lower():
                    return d
            for d in all_subdirs:
                if pat_lower in d.name.lower():
                    return d
        return None

    # Identify Train dirs
    train_img_dir = _find_dir_by_pattern([
        "ISIC2018_Task1-1_Training_Input",
        "Task1-1_Training_Input",
        "Training_Input",
        "train_images",
        "train/images",
        "train",
    ])
    train_mask_dir = _find_dir_by_pattern([
        "ISIC2018_Task1_Training_GroundTruth",
        "Training_GroundTruth",
        "train_masks",
        "train/masks",
        "GroundTruth",
    ])

    # Identify Validation dirs
    val_img_dir = _find_dir_by_pattern([
        "ISIC2018_Task1-2_Validation_Input",
        "Task1-2_Validation_Input",
        "Validation_Input",
        "val_images",
        "val/images",
        "val",
        "validation",
    ])
    val_mask_dir = _find_dir_by_pattern([
        "ISIC2018_Task1_Validation_GroundTruth",
        "Validation_GroundTruth",
        "val_masks",
        "val/masks",
    ])

    # Identify Test dirs
    test_img_dir = _find_dir_by_pattern([
        "ISIC2018_Task1-3_Test_Input",
        "Task1-3_Test_Input",
        "Test_Input",
        "test_images",
        "test/images",
        "test",
    ])
    test_mask_dir = _find_dir_by_pattern([
        "ISIC2018_Task1_Test_GroundTruth",
        "Test_GroundTruth",
        "test_masks",
        "test/masks",
    ])

    # Build pairs
    train_pairs, train_orphans, _ = _match_pairs(train_img_dir, train_mask_dir)
    val_pairs, val_orphans, _ = _match_pairs(val_img_dir, val_mask_dir)
    test_pairs, test_orphans, _ = _match_pairs(test_img_dir, test_mask_dir)

    # If dataset has images in root or a single flat folder
    if len(train_pairs) == 0 and len(val_pairs) == 0:
        # Check flat images / masks in root
        images_in_root = [d for d in all_subdirs if "image" in d.name.lower()]
        masks_in_root = [d for d in all_subdirs if "mask" in d.name.lower() or "groundtruth" in d.name.lower()]
        if images_in_root and masks_in_root:
            all_pairs, _, _ = _match_pairs(images_in_root[0], masks_in_root[0])
            train_pairs = all_pairs

    stats = {
        "dataset_root": str(root_path),
        "total_train": len(train_pairs),
        "total_val": len(val_pairs),
        "total_test": len(test_pairs),
        "train_image_dir": str(train_img_dir) if train_img_dir else None,
        "train_mask_dir": str(train_mask_dir) if train_mask_dir else None,
        "val_image_dir": str(val_img_dir) if val_img_dir else None,
        "val_mask_dir": str(val_mask_dir) if val_mask_dir else None,
        "test_image_dir": str(test_img_dir) if test_img_dir else None,
        "test_mask_dir": str(test_mask_dir) if test_mask_dir else None,
        "train_orphans_count": len(train_orphans),
        "val_orphans_count": len(val_orphans),
        "test_orphans_count": len(test_orphans),
    }

    return {
        "root": str(root_path),
        "exists": True,
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
        "stats": stats,
    }
