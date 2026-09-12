"""Cross-check offline inference pipeline vs live HTTP API inference."""

import base64
import io
import json
import urllib.request
from pathlib import Path
from PIL import Image
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infer import run_inference

def cross_check(image_path: str):
    print("=" * 70)
    print(f"CROSS-CHECK AUDIT: {image_path}")
    print("=" * 70)

    # 1. OFFLINE INFERENCE
    print("\n--- [A] Running OFFLINE Pipeline (infer.py) ---")
    offline_res = run_inference(
        image_path=image_path,
        checkpoint_path="outputs/checkpoints/full_rb_unet_latest.pth",
        output_dir="outputs/qualitative/offline_audit",
    )
    offline_mask = np.array(Image.open(offline_res["mask_path"])) > 127
    offline_area_pct = offline_res["lesion_area_pct"]

    # 2. LIVE HTTP API INFERENCE
    print("\n--- [B] Running LIVE API Pipeline (POST /api/segment) ---")
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    req = urllib.request.Request(
        "http://localhost:8000/api/segment",
        data=json.dumps({"image": img_b64}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        live_data = json.loads(resp.read().decode("utf-8"))

    assert live_data["success"], f"Live API returned failure: {live_data}"
    live_area_pct = live_data["lesion_area_pct"]

    # Decode live mask
    live_mask_b64 = live_data["mask"].split(",", 1)[1]
    live_mask_img = Image.open(io.BytesIO(base64.b64decode(live_mask_b64)))
    live_mask = np.array(live_mask_img) > 127

    # 3. COMPARISON
    print("\n--- [C] Numerical and Spatial Comparison ---")
    print(f"Offline Mask Shape: {offline_mask.shape}")
    print(f"Live Mask Shape:    {live_mask.shape}")
    print(f"Offline Lesion Area: {offline_area_pct:.2f}%")
    print(f"Live Lesion Area:    {live_area_pct:.2f}%")

    area_diff = abs(offline_area_pct - live_area_pct)
    print(f"Lesion Area Delta:   {area_diff:.3f}%")

    # If shapes match, compute IoU between masks
    if offline_mask.shape == live_mask.shape:
        intersection = np.logical_and(offline_mask, live_mask).sum()
        union = np.logical_or(offline_mask, live_mask).sum()
        mask_iou = intersection / union if union > 0 else 1.0
        print(f"Offline vs Live Mask Agreement (IoU): {mask_iou:.4f}")
        assert mask_iou > 0.98, f"Mask agreement too low: {mask_iou:.4f}"
    else:
        # If live scaled down for display, scale offline mask to same display dimensions and test IoU
        disp_h, disp_w = live_mask.shape
        offline_scaled = np.array(Image.fromarray(offline_mask.astype(np.uint8) * 255).resize((disp_w, disp_h), resample=Image.Resampling.NEAREST)) > 127
        intersection = np.logical_and(offline_scaled, live_mask).sum()
        union = np.logical_or(offline_scaled, live_mask).sum()
        mask_iou = intersection / union if union > 0 else 1.0
        print(f"Offline vs Live Resampled Mask Agreement (IoU): {mask_iou:.4f}")
        assert mask_iou > 0.95, f"Mask agreement too low: {mask_iou:.4f}"

    assert area_diff < 0.5, f"Lesion area discrepancy too large: {area_diff}"
    print("\n>>> VALIDATION PASSED: Offline and Live pipelines are numerically and spatially equivalent!")
    print("=" * 70)

if __name__ == "__main__":
    cross_check("data/raw/ISIC2018_Task1-2_Validation_Input/ISIC_0012255.jpg")
