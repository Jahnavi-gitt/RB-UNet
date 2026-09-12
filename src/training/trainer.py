"""Unified trainer engine for baseline U-Net and RB-UNet."""

import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruption.corruptions import apply_random_corruption
from src.evaluation.metrics import evaluate_batch_metrics
from src.losses.boundary_loss import BoundaryLoss
from src.losses.consistency_loss import ConsistencyLoss
from src.losses.segmentation_loss import CombinedSegmentationLoss
from src.models.boundary_head import extract_boundary_targets
from .checkpoint import save_checkpoint


class Trainer:
    """Orchestrates model training, validation, checkpointing, and metrics tracking."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_dir: str = "outputs/checkpoints",
        experiment_name: str = "experiment",
        scheduler: Optional[Any] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.scheduler = scheduler

        # Loss configurations
        loss_cfg = config.get("loss", {})
        self.dice_weight = float(loss_cfg.get("dice_weight", 1.0))
        self.bce_weight = float(loss_cfg.get("bce_weight", 1.0))
        self.boundary_weight = float(loss_cfg.get("boundary_weight", 0.5))
        self.consistency_weight = float(loss_cfg.get("consistency_weight", 0.5))

        # Instantiate losses
        self.seg_loss_fn = CombinedSegmentationLoss(
            dice_weight=self.dice_weight,
            bce_weight=self.bce_weight,
            eps=float(loss_cfg.get("epsilon", 1e-6)),
        ).to(device)
        self.boundary_loss_fn = BoundaryLoss().to(device)
        self.consistency_loss_fn = ConsistencyLoss().to(device)

        self.best_val_dice = -1.0
        self.history = []

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            out = self.model(images)
            if isinstance(out, dict):
                clean_seg_logits = out["seg_logits"]
                boundary_logits = out.get("boundary_logits")
            else:
                clean_seg_logits = out
                boundary_logits = None

            # 1. Primary Segmentation Loss
            loss = self.seg_loss_fn(clean_seg_logits, masks)

            # 2. Auxiliary Boundary Loss (if model supports it and weight > 0)
            if boundary_logits is not None and self.boundary_weight > 0.0:
                boundary_targets = extract_boundary_targets(masks).to(self.device)
                bnd_loss = self.boundary_loss_fn(boundary_logits, boundary_targets)
                loss = loss + self.boundary_weight * bnd_loss

            # 3. Corruption Consistency Loss (if enabled and weight > 0)
            if self.consistency_weight > 0.0:
                corrupted_images, _ = apply_random_corruption(images)
                corrupted_images = corrupted_images.to(self.device)
                out_corrupt = self.model(corrupted_images)
                corrupt_seg_logits = out_corrupt["seg_logits"] if isinstance(out_corrupt, dict) else out_corrupt
                const_loss = self.consistency_loss_fn(clean_seg_logits, corrupt_seg_logits)
                loss = loss + self.consistency_weight * const_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        dice_scores = []
        iou_scores = []

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            out = self.model(images)
            seg_logits = out["seg_logits"] if isinstance(out, dict) else out

            metrics = evaluate_batch_metrics(seg_logits, masks)
            dice_scores.append(metrics["dice"])
            iou_scores.append(metrics["iou"])

        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
        return {"val_dice": mean_dice, "val_iou": mean_iou}

    def fit(self, epochs: int) -> Dict[str, Any]:
        print(f"\n[Training] Starting experiment '{self.experiment_name}' for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()
            val_dice = val_metrics["val_dice"]
            val_iou = val_metrics["val_iou"]

            if self.scheduler:
                self.scheduler.step()

            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice

            # Save checkpoint
            ckpt_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pth"
            save_checkpoint(
                model=self.model,
                filepath=str(ckpt_path),
                epoch=epoch,
                val_dice=val_dice,
                val_iou=val_iou,
                optimizer=self.optimizer,
                config=self.config,
                is_best=is_best,
            )

            epoch_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "is_best": is_best,
            }
            self.history.append(epoch_log)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Dice: {val_dice:.4f} | "
                f"Val IoU: {val_iou:.4f}"
                f"{' [BEST]' if is_best else ''}"
            )

        elapsed = time.time() - start_time
        print(f"[Training] Completed in {elapsed:.1f}s. Best Val Dice: {self.best_val_dice:.4f}")
        return {
            "best_val_dice": self.best_val_dice,
            "history": self.history,
            "elapsed_seconds": elapsed,
        }
