"""Single-image inference visualizer and prediction engine."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.rb_unet import RBUNet
from src.models.unet import UNet
from src.training.checkpoint import load_checkpoint
from src.utils.visualization import create_overlay, plot_comparative_panel


def clean_isolated_artifacts(mask_bin: np.ndarray, min_size: int = 20) -> np.ndarray:
    """Removes isolated stray prediction flecks while preserving real multi-component lesions."""
    import scipy.ndimage as ndi
    labeled, num_features = ndi.label(mask_bin > 0)
    if num_features <= 1:
        return mask_bin
    counts = np.bincount(labeled.ravel())
    clean_mask = np.zeros_like(mask_bin)
    for idx in range(1, num_features + 1):
        if counts[idx] >= min_size:
            clean_mask[labeled == idx] = 1
    if clean_mask.sum() == 0 and mask_bin.sum() > 0:
        largest_idx = np.argmax(counts[1:]) + 1
        clean_mask[labeled == largest_idx] = 1
    return clean_mask


def draw_boundary_contour(image_np: np.ndarray, mask_bin: np.ndarray, color=(14, 165, 233), thickness: int = 2) -> np.ndarray:
    """Renders a crisp boundary contour directly on the RGB image."""
    from scipy.ndimage import binary_dilation
    struct = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(mask_bin > 0, structure=struct, iterations=thickness)
    contour = dilated ^ (mask_bin > 0)
    out = image_np.copy()
    out[contour] = color
    return out


def run_inference(
    image_path: str,
    checkpoint_path: Optional[str] = None,
    output_dir: str = "outputs/qualitative",
    device_name: str = "auto",
    image_size: int = 256,
    threshold: float = 0.70,
):
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    # Load and preprocess input image
    raw_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = raw_img.size
    resized = TF.resize(raw_img, [image_size, image_size], interpolation=TF.InterpolationMode.BILINEAR)
    img_tensor = TF.to_tensor(resized)
    norm_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)

    # Instantiate model
    features = [16, 32, 64, 128, 256]
    if checkpoint_path and os.path.exists(checkpoint_path):
        from src.training.checkpoint import build_model_from_checkpoint
        model = build_model_from_checkpoint(checkpoint_path, device=str(device))
    else:
        # Default fallback RB-UNet if checkpoint is omitted
        model = RBUNet(in_channels=3, out_channels=1, features=features).to(device)

    model.eval()

    # Time the inference
    t0 = time.time()
    with torch.no_grad():
        out = model(norm_tensor)
    inference_time_ms = (time.time() - t0) * 1000.0

    if isinstance(out, dict):
        seg_logits = out["seg_logits"]
        bnd_logits = out.get("boundary_logits")
    else:
        seg_logits = out
        bnd_logits = None

    prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
    pred_mask_256 = (prob > threshold).astype(np.uint8)
    pred_mask_256 = clean_isolated_artifacts(pred_mask_256, min_size=20)

    # Resize mask back to ORIGINAL image dimensions using NEAREST interpolation
    mask_pil_256 = Image.fromarray(pred_mask_256 * 255)
    mask_pil_orig = mask_pil_256.resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
    pred_mask = (np.array(mask_pil_orig) > 127).astype(np.uint8)

    # Compute genuine lesion area % on original dimensions
    lesion_pixels = int(np.sum(pred_mask))
    total_pixels = pred_mask.size
    lesion_area_pct = (lesion_pixels / total_pixels) * 100.0

    # Boundary map resized back to original dimensions
    bnd_map = None
    if bnd_logits is not None:
        bnd_prob = torch.sigmoid(bnd_logits).squeeze().cpu().numpy()
        bnd_mask_256 = (bnd_prob > threshold).astype(np.uint8)
        bnd_pil_orig = Image.fromarray(bnd_mask_256 * 255).resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
        bnd_map = (np.array(bnd_pil_orig) > 127).astype(np.uint8)

    # Generate overlay and contour on the original RGB image (pixel-aligned)
    img_np = np.array(raw_img)
    overlay_np = create_overlay(img_np, pred_mask, color=(14, 165, 233), alpha=0.45)
    contour_np = draw_boundary_contour(img_np, pred_mask, color=(14, 165, 233), thickness=2)

    # Save output artifacts
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    mask_file = out_path / f"{stem}_predicted_mask.png"
    overlay_file = out_path / f"{stem}_overlay.png"
    panel_file = out_path / f"{stem}_comparison.png"

    Image.fromarray(pred_mask * 255).save(mask_file)
    Image.fromarray(overlay_np).save(overlay_file)

    # For panel comparison, use display-sized version to fit nicely in figure
    disp_size = (512, int(512 * orig_h / orig_w))
    img_disp = np.array(raw_img.resize(disp_size, resample=Image.Resampling.BILINEAR))
    mask_disp = np.array(mask_pil_orig.resize(disp_size, resample=Image.Resampling.NEAREST))
    bnd_disp = np.array(Image.fromarray(bnd_map).resize(disp_size, resample=Image.Resampling.NEAREST)) if bnd_map is not None else None

    plot_comparative_panel(
        image=img_disp,
        ground_truth=None,
        baseline_pred=None,
        rb_unet_pred=mask_disp,
        boundary_pred=bnd_disp,
        save_path=str(panel_file),
        title=f"Inference: {stem} | Lesion Area: {lesion_area_pct:.1f}% | Time: {inference_time_ms:.1f}ms",
    )

    print(f"\n[Inference Result for {stem}]")
    print(f"  Original Size:      {orig_w}x{orig_h}")
    print(f"  Lesion Area:        {lesion_area_pct:.2f}%")
    print(f"  Inference Time:     {inference_time_ms:.1f} ms")
    print(f"  Predicted Mask:     {mask_file}")
    print(f"  Overlay Image:      {overlay_file}")
    print(f"  Comparative Panel:  {panel_file}")

    return {
        "sample_id": stem,
        "lesion_area_pct": lesion_area_pct,
        "inference_time_ms": inference_time_ms,
        "mask_path": str(mask_file),
        "overlay_path": str(overlay_file),
        "panel_path": str(panel_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lesion segmentation inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (JPG/PNG)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs/qualitative")
    args = parser.parse_args()

    run_inference(image_path=args.image, checkpoint_path=args.checkpoint, output_dir=args.output_dir)
