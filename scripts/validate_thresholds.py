import torch
import numpy as np
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.split_discovery import discover_dataset_splits
from src.datasets.transforms import get_transforms
from src.datasets.isic_dataset import ISICDataset
from torch.utils.data import DataLoader
from src.training.checkpoint import build_model_from_checkpoint
from src.evaluation.metrics import compute_dice, compute_iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discovered = discover_dataset_splits("data/raw")
val_samples = discovered["val"]
print(f"Discovered {len(val_samples)} validation samples.")

val_tf = get_transforms(image_size=256, is_training=False)
val_ds = ISICDataset(val_samples, transform=val_tf)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
sweep_results = {}

for name, ckpt_file in [("Baseline U-Net", "outputs/checkpoints/baseline_unet_latest.pth"), ("RB-UNet", "outputs/checkpoints/full_rb_unet_latest.pth")]:
    print("="*60)
    print(f"EVALUATING {name}: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    print(f"Epoch: {ckpt.get('epoch')}, Checkpoint Val Dice: {ckpt.get('val_dice')}")
    
    model = build_model_from_checkpoint(ckpt_file, device=str(device))
    model.eval()
    
    sweep_results[name] = []
    best_th = 0.5
    best_dice = -1.0
    best_iou = -1.0
    
    for th in thresholds:
        dices, ious = [], []
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                target = batch["mask"].to(device)
                out = model(img)
                logits = out["seg_logits"] if isinstance(out, dict) else out
                prob = torch.sigmoid(logits)
                d = compute_dice(prob, target, threshold=th)
                u = compute_iou(prob, target, threshold=th)
                dices.append(d)
                ious.append(u)
        mean_dice = float(np.mean(dices))
        mean_iou = float(np.mean(ious))
        sweep_results[name].append({"threshold": round(th, 2), "val_dice": round(mean_dice, 4), "val_iou": round(mean_iou, 4)})
        print(f"Threshold {th:.2f} -> Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f}")
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_iou = mean_iou
            best_th = th
            
    print(f"BEST THRESHOLD for {name}: {best_th:.2f} (Val Dice: {best_dice:.4f}, IoU: {best_iou:.4f})")

Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
with open("outputs/metrics/validation_threshold_sweep.json", "w") as f:
    json.dump(sweep_results, f, indent=2)
print("Saved to outputs/metrics/validation_threshold_sweep.json")
