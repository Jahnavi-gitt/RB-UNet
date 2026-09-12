"""Master end-to-end integration audit for RB-UNet system."""

import base64
import json
import urllib.request
from pathlib import Path


def run_full_system_verification():
    print("=" * 70)
    print("RB-UNET END-TO-END SYSTEM INTEGRATION AUDIT")
    print("=" * 70)

    # 1. API Status
    st_req = urllib.request.urlopen("http://localhost:8000/api/status")
    st_res = json.loads(st_req.read().decode())
    print("\n[1] API Status Check:")
    for k, v in st_res.items():
        print(f"    - {k:25}: {v}")
    assert st_res["best_checkpoint_exists"], "Best checkpoint missing"
    assert st_res["baseline_checkpoint_exists"], "Baseline checkpoint missing"

    # 2. Research Results
    res_req = urllib.request.urlopen("http://localhost:8000/api/results")
    res_data = json.loads(res_req.read().decode())
    print("\n[2] Research Results Contract:")
    print(f"    - Ablation stages: {len(res_data.get('ablation', []))}")
    print(f"    - Robustness cohorts: {len(res_data.get('robustness', []))}")
    print(f"    - Sacred test set evaluated: {bool(res_data.get('final_test'))}")
    print(f"    - Qualitative cases: {len(res_data.get('qualitative', []))}")

    # 3. Live Segmentation across 5 real categories
    presets = [
        ("small_lesion", "Small Lesion"),
        ("large_lesion", "Large Lesion"),
        ("irregular_boundary", "Irregular Boundary"),
        ("low_contrast", "Low Contrast"),
        ("challenging_example", "Challenging Example"),
    ]
    print("\n[3] Live Segmentation (/api/segment) on Real Samples:")
    for pid, pname in presets:
        img_path = Path(f"outputs/sample_images/{pid}.jpg")
        with open(img_path, "rb") as f:
            b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
        req = urllib.request.Request(
            "http://localhost:8000/api/segment",
            data=json.dumps({"image": b64}).encode(),
            headers={"Content-Type": "application/json"},
        )
        res = json.loads(urllib.request.urlopen(req).read().decode())
        area = res["lesion_area_pct"]
        th = res["threshold"]
        fg_prob = res["mean_foreground_prob"]
        ms = res["inference_time_ms"]
        print(f"    - {pname:22} | Area: {area:5.1f}% | Th: {th:.2f} | Mean FG Prob: {fg_prob:.3f} | Latency: {ms:5.1f}ms")
        assert res["success"], "Segmentation failed"
        assert res["mask"] and res["overlay"] and res["contour"], "Visual outputs missing"

    # 4. Live Robustness on small_lesion across all 5 conditions
    print("\n[4] Live Robustness (/api/robustness) on Active Image under Perturbations:")
    with open("outputs/sample_images/small_lesion.jpg", "rb") as f:
        b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    conditions = ["Clean", "Gaussian Noise", "Gaussian Blur", "Contrast Shift", "Brightness Shift"]
    for c in conditions:
        req = urllib.request.Request(
            "http://localhost:8000/api/robustness",
            data=json.dumps({"image": b64, "condition": c}).encode(),
            headers={"Content-Type": "application/json"},
        )
        res = json.loads(urllib.request.urlopen(req).read().decode())
        b_area = res["unet_area_pct"]
        rb_area = res["rb_unet_area_pct"]
        consistency = res["prediction_consistency"]
        delta_area = res["area_delta_pct"]
        print(f"    - {c:18} | Base Area: {b_area:5.1f}% | RB Area: {rb_area:5.1f}% | Consistency: {consistency:5.1f}% | Delta Area: {delta_area:4.1f}%")
        assert res["success"], "Robustness failed"
        assert res["clean_image"] and res["degraded_image"] and res["unet_mask"] and res["rb_unet_mask"]

    print("\n" + "=" * 70)
    print("ALL AUDIT VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_system_verification()
