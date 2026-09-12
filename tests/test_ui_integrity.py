"""Test UI DOM consistency and static asset existence."""

import re
from pathlib import Path
import pytest


def test_html_and_js_id_integrity():
    html = Path("ui/index.html").read_text(encoding="utf-8")
    js = Path("ui/js/app.js").read_text(encoding="utf-8")

    html_ids = set(re.findall(r'id=["\']([^"\']+)["\']', html))
    js_ids = set(re.findall(r'getElementById\(["\']([^"\']+)["\']\)', js))

    missing = js_ids - html_ids
    assert len(missing) == 0, f"JS references IDs not found in HTML: {missing}"


def test_robustness_image_artifacts_exist():
    rob_dir = Path("outputs/robustness")
    assert rob_dir.exists(), "outputs/robustness directory must exist"

    conditions = ["clean", "gaussian_noise", "gaussian_blur", "contrast", "brightness"]
    for c in conditions:
        cond_dir = rob_dir / c
        assert cond_dir.exists(), f"Condition directory {c} must exist"
        assert (cond_dir / "input.png").exists(), f"input.png missing in {c}"
        assert (cond_dir / "baseline_prediction.png").exists(), f"baseline_prediction.png missing in {c}"
        assert (cond_dir / "rb_unet_prediction.png").exists(), f"rb_unet_prediction.png missing in {c}"
        assert (cond_dir / "input.png").stat().st_size > 0
        assert (cond_dir / "baseline_prediction.png").stat().st_size > 0
        assert (cond_dir / "rb_unet_prediction.png").stat().st_size > 0


def test_sample_presets_exist():
    sample_dir = Path("outputs/sample_images")
    assert sample_dir.exists(), "outputs/sample_images directory must exist"
    for i in range(1, 4):
        sample = sample_dir / f"sample{i}.jpg"
        assert sample.exists(), f"sample{i}.jpg missing"
        assert sample.stat().st_size > 0


def test_trained_checkpoints_exist():
    ckpt_dir = Path("outputs/checkpoints")
    assert (ckpt_dir / "baseline_unet_latest.pth").exists()
    assert (ckpt_dir / "full_rb_unet_latest.pth").exists()
    assert (ckpt_dir / "full_rb_unet_latest.pth").stat().st_size > 1_000_000
