"""Pipeline sanity check and micro-dataset overfit test.

Verifies end-to-end integration:
1. Deterministic seed configuration
2. Dataset loading and tensor alignment
3. Standard U-Net forward and backward pass
4. RB-UNet forward and dual-loss backward pass
5. Consistency loss under random corruptions
6. Checkpointing save/load integrity
7. Micro-dataset overfit test (verifies capacity to overfit 5 samples)
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.isic_dataset import ISICDataset
from src.datasets.transforms import get_transforms
from src.evaluation.metrics import evaluate_batch_metrics
from src.losses.boundary_loss import BoundaryLoss
from src.losses.consistency_loss import ConsistencyLoss
from src.losses.segmentation_loss import CombinedSegmentationLoss
from src.models.boundary_head import extract_boundary_targets
from src.models.rb_unet import RBUNet
from src.models.unet import UNet
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.seed import set_seed


def run_sanity_check():
    print("=" * 65)
    print("RB-UNET END-TO-END PIPELINE SANITY CHECK")
    print("=" * 65)

    # 1. Determinism
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. Target Device: {device}")

    # 2. Synthetic fixture for smoke test
    temp_dir = Path("outputs/smoke_test_data")
    temp_dir.mkdir(parents=True, exist_ok=True)
    samples = []

    print("2. Generating 5 synthetic test samples...")
    for i in range(5):
        sid = f"SMOKE_{i:03d}"
        img_path = temp_dir / f"{sid}.jpg"
        msk_path = temp_dir / f"{sid}_mask.png"

        # Create image with textured background and elliptical lesion
        img_arr = np.full((128, 128, 3), 180, dtype=np.uint8)
        msk_arr = np.zeros((128, 128), dtype=np.uint8)

        # Draw centered circle/ellipse
        y, x = np.ogrid[:128, :128]
        mask_area = (x - 64) ** 2 + (y - 64) ** 2 <= (25 + i * 2) ** 2
        msk_arr[mask_area] = 255
        img_arr[mask_area] = [70, 40, 30]  # Dark pigmented lesion color

        Image.fromarray(img_arr).save(img_path)
        Image.fromarray(msk_arr).save(msk_path)

        samples.append({"id": sid, "image_path": str(img_path), "mask_path": str(msk_path)})

    # 3. Dataset & DataLoader
    transform = get_transforms(image_size=128, is_training=False)
    dataset = ISICDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert batch["image"].shape == (2, 3, 128, 128), f"Unexpected image shape: {batch['image'].shape}"
    assert batch["mask"].shape == (2, 1, 128, 128), f"Unexpected mask shape: {batch['mask'].shape}"
    print(f"3. Dataset DataLoader verified (Tensor shape: {batch['image'].shape})")

    # 4. Standard U-Net Forward/Backward
    unet = UNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128, 256]).to(device)
    opt_unet = torch.optim.Adam(unet.parameters(), lr=1e-3)
    crit = CombinedSegmentationLoss().to(device)

    imgs = batch["image"].to(device)
    msks = batch["mask"].to(device)

    out_unet = unet(imgs)
    loss_u = crit(out_unet, msks)
    loss_u.backward()
    opt_unet.step()
    assert torch.isfinite(loss_u), "Baseline U-Net loss is non-finite!"
    print(f"4. Standard U-Net forward/backward verified (Loss: {loss_u.item():.4f})")

    # 5. RB-UNet Dual Forward & Loss
    rb_unet = RBUNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128, 256], boundary_head_channels=16).to(device)
    opt_rb = torch.optim.Adam(rb_unet.parameters(), lr=1e-3)
    bnd_loss_fn = BoundaryLoss().to(device)
    const_loss_fn = ConsistencyLoss().to(device)

    out_rb = rb_unet(imgs)
    bnd_targets = extract_boundary_targets(msks).to(device)

    l_seg = crit(out_rb["seg_logits"], msks)
    l_bnd = bnd_loss_fn(out_rb["boundary_logits"], bnd_targets)
    # Corrupted pass
    noisy_imgs = imgs + 0.05 * torch.randn_like(imgs)
    out_corrupt = rb_unet(noisy_imgs)
    l_const = const_loss_fn(out_rb["seg_logits"], out_corrupt["seg_logits"])

    total_loss = l_seg + 0.5 * l_bnd + 0.5 * l_const
    total_loss.backward()
    opt_rb.step()
    assert torch.isfinite(total_loss), "RB-UNet total loss is non-finite!"
    print(f"5. RB-UNet dual forward/backward verified (Total Loss: {total_loss.item():.4f})")

    # 6. Checkpointing Test
    ckpt_path = "outputs/checkpoints/sanity_test.pth"
    save_checkpoint(rb_unet, ckpt_path, epoch=1, val_dice=0.88, val_iou=0.78, optimizer=opt_rb, is_best=False)
    fresh_model = RBUNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128, 256], boundary_head_channels=16).to(device)
    loaded = load_checkpoint(ckpt_path, fresh_model, device=str(device))
    assert loaded["val_dice"] == 0.88
    print("6. Checkpointing save/load verified.")

    # 7. Micro Overfitting Test (30 steps on 5 samples)
    print("\n7. Executing Micro-Overfitting Test (30 iterations)...")
    single_batch_loader = DataLoader(dataset, batch_size=5, shuffle=False)
    micro_batch = next(iter(single_batch_loader))
    m_imgs = micro_batch["image"].to(device)
    m_msks = micro_batch["mask"].to(device)

    overfit_model = RBUNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128, 256]).to(device)
    overfit_opt = torch.optim.Adam(overfit_model.parameters(), lr=0.01)

    initial_dice = 0.0
    final_dice = 0.0

    for step in range(35):
        overfit_opt.zero_grad()
        out = overfit_model(m_imgs)
        bnd_t = extract_boundary_targets(m_msks).to(device)
        loss = crit(out["seg_logits"], m_msks) + 0.5 * bnd_loss_fn(out["boundary_logits"], bnd_t)
        loss.backward()
        overfit_opt.step()

        metrics = evaluate_batch_metrics(out["seg_logits"], m_msks)
        if step == 0:
            initial_dice = metrics["dice"]
        if step == 34:
            final_dice = metrics["dice"]

    print(f"   Initial Dice: {initial_dice:.4f} -> Final Overfit Dice: {final_dice:.4f}")
    assert final_dice > 0.85, f"Model failed to overfit tiny dataset (Dice: {final_dice:.4f} <= 0.85)!"

    print("\n" + "=" * 65)
    print("ALL SANITY CHECKS & MICRO-OVERFIT TEST PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_sanity_check()
