"""Generate real qualitative research comparison figures using trained models."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

from src.datasets.split_discovery import discover_dataset_splits
from src.training.checkpoint import build_model_from_checkpoint
from src.evaluation.metrics import compute_dice, compute_iou

def main():
    device = torch.device("cpu")
    out_dir = Path("outputs/qualitative/comparison_suite")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model = build_model_from_checkpoint("outputs/checkpoints/baseline_unet_latest.pth", device="cpu")
    rb_model = build_model_from_checkpoint("outputs/checkpoints/full_rb_unet_latest.pth", device="cpu")
    base_model.eval()
    rb_model.eval()

    splits = discover_dataset_splits("data/raw")
    val_samples = {s["id"]: s for s in splits["val"]}

    # Selected representative cases
    cases = [
        {"id": "ISIC_0012255", "label": "Small Focal Lesion", "difficulty": "High - Subtle scale"},
        {"id": "ISIC_0012585", "label": "Irregular Scalloped Margin", "difficulty": "High - Boundary complexity"},
        {"id": "ISIC_0015370", "label": "Challenging Asymmetric Border", "difficulty": "Moderate - Multi-lobular"},
        {"id": "ISIC_0015480", "label": "Low Contrast Lesion", "difficulty": "High - Faint gradient"},
    ]

    manifest = []

    for c in cases:
        sid = c["id"]
        if sid not in val_samples:
            continue
        meta = val_samples[sid]
        raw_img = Image.open(meta["image_path"]).convert("RGB")
        raw_gt = Image.open(meta["mask_path"]).convert("L")
        w, h = raw_img.size

        # Preprocessing matching training
        resized = TF.resize(raw_img, [256, 256], interpolation=TF.InterpolationMode.BILINEAR)
        t_img = TF.to_tensor(resized)
        norm = TF.normalize(t_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)

        # Baseline inference
        with torch.no_grad():
            out_base = base_model(norm)
        prob_base = torch.sigmoid(out_base).squeeze().numpy()
        mask_base_256 = (prob_base > 0.5).astype(np.uint8)
        mask_base_orig = Image.fromarray(mask_base_256 * 255).resize((w, h), resample=Image.Resampling.NEAREST)

        # RB-UNet inference
        with torch.no_grad():
            out_rb = rb_model(norm)
        prob_rb = torch.sigmoid(out_rb["seg_logits"]).squeeze().numpy()
        mask_rb_256 = (prob_rb > 0.5).astype(np.uint8)
        mask_rb_orig = Image.fromarray(mask_rb_256 * 255).resize((w, h), resample=Image.Resampling.NEAREST)

        # Quantitative scores on 256 standard evaluation grid
        gt_256 = (TF.to_tensor(TF.resize(raw_gt, [256, 256], interpolation=TF.InterpolationMode.NEAREST)) > 0.5).float()
        base_dice = compute_dice(torch.from_numpy(prob_base).unsqueeze(0).unsqueeze(0), gt_256, threshold=0.5)
        base_iou = compute_iou(torch.from_numpy(prob_base).unsqueeze(0).unsqueeze(0), gt_256, threshold=0.5)
        rb_dice = compute_dice(torch.from_numpy(prob_rb).unsqueeze(0).unsqueeze(0), gt_256, threshold=0.5)
        rb_iou = compute_iou(torch.from_numpy(prob_rb).unsqueeze(0).unsqueeze(0), gt_256, threshold=0.5)

        # Save display-optimized images (max 600px for table gallery viewing)
        disp_w = 480
        disp_h = int(480 * h / w)

        p_orig = out_dir / f"{sid}_orig.jpg"
        p_gt = out_dir / f"{sid}_gt.png"
        p_base = out_dir / f"{sid}_base.png"
        p_rb = out_dir / f"{sid}_rb.png"

        raw_img.resize((disp_w, disp_h), resample=Image.Resampling.BILINEAR).save(p_orig, quality=90)
        raw_gt.resize((disp_w, disp_h), resample=Image.Resampling.NEAREST).save(p_gt)
        mask_base_orig.resize((disp_w, disp_h), resample=Image.Resampling.NEAREST).save(p_base)
        mask_rb_orig.resize((disp_w, disp_h), resample=Image.Resampling.NEAREST).save(p_rb)

        manifest.append({
            "id": sid,
            "label": c["label"],
            "difficulty": c["difficulty"],
            "original_url": f"/outputs/qualitative/comparison_suite/{sid}_orig.jpg",
            "gt_url": f"/outputs/qualitative/comparison_suite/{sid}_gt.png",
            "baseline_url": f"/outputs/qualitative/comparison_suite/{sid}_base.png",
            "rb_unet_url": f"/outputs/qualitative/comparison_suite/{sid}_rb.png",
            "baseline_dice": round(base_dice, 4),
            "baseline_iou": round(base_iou, 4),
            "rb_unet_dice": round(rb_dice, 4),
            "rb_unet_iou": round(rb_iou, 4),
            "delta_dice": round(rb_dice - base_dice, 4),
        })
        print(f"Generated qualitative case: {sid} ({c['label']}) | Base Dice: {base_dice:.4f}, RB Dice: {rb_dice:.4f}")

    with open(out_dir / "qualitative_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Qualitative suite generated successfully.")

if __name__ == "__main__":
    main()
