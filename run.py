"""Master one-command end-to-end orchestration pipeline for RB-UNet."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.inspect_dataset import inspect_dataset
from scripts.sanity_check import run_sanity_check
from train import run_training
from scripts.robustness_test import run_robustness_test
from scripts.final_test import run_final_test


def execute_pipeline(
    config_path: str = "configs/config.yaml",
    smoke_test: bool = False,
    serve_ui: bool = False,
):
    print("=" * 75)
    print("RB-UNET: ROBUST BOUNDARY-CONSISTENT U-NET — MASTER PIPELINE")
    print("=" * 75)
    start_total_time = time.time()

    # 1. Dataset Audit
    print("\n>>> [STAGE 1/6] Dataset Audit & Split Discovery")
    dataset_report = inspect_dataset(config_path=config_path)

    # 2. Pipeline Sanity Check
    print("\n>>> [STAGE 2/6] Pipeline Sanity & Micro-Overfit Smoke Test")
    run_sanity_check()

    # 3. Model Training & Ablations
    print("\n>>> [STAGE 3/6] Controlled Training & Ablation Suite")
    ablations = [
        ("configs/baseline.yaml", "baseline_unet", "Experiment 1: Standard U-Net Baseline"),
        ("configs/boundary.yaml", "ablation_boundary", "Experiment 2: U-Net + Boundary Supervision"),
        ("configs/consistency.yaml", "ablation_consistency", "Experiment 3: U-Net + Corruption Consistency"),
        ("configs/rb_unet.yaml", "full_rb_unet", "Experiment 4: Full Proposed RB-UNet"),
    ]

    ablation_results = []
    epochs_per_run = 2 if smoke_test else None

    for cfg_file, exp_id, title in ablations:
        print(f"\n--- {title} ---")
        summary = run_training(
            config_path=cfg_file,
            experiment_name=exp_id,
            smoke_test=smoke_test,
            override_epochs=epochs_per_run,
        )
        ablation_results.append(summary)

    # Compile ablation summary
    ablation_dir = Path("outputs/ablations")
    ablation_dir.mkdir(parents=True, exist_ok=True)
    with open(ablation_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    # 4. Robustness Benchmark
    print("\n>>> [STAGE 4/6] Robustness Stress-Testing (Clean vs Degraded)")
    robustness_results = run_robustness_test(
        baseline_ckpt="outputs/checkpoints/baseline_unet_latest.pth",
        rb_ckpt="outputs/checkpoints/full_rb_unet_latest.pth",
        config_path=config_path,
        smoke_test=smoke_test,
    )

    # 5. Final Sacred Test Evaluation
    print("\n>>> [STAGE 5/6] Final Sacred Test Set Evaluation")
    final_test_results = run_final_test(
        baseline_ckpt="outputs/checkpoints/baseline_unet_latest.pth",
        rb_ckpt="outputs/checkpoints/full_rb_unet_latest.pth",
        config_path=config_path,
        smoke_test=smoke_test,
    )

    # 6. Qualitative Demo Figure Generation
    print("\n>>> [STAGE 6/6] Generating Demo & UI Visual Artifacts")
    from infer import run_inference
    demo_dir = Path("outputs/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Run inference on sample image
    sample_images = list(Path("outputs/smoke_test_data").glob("*.jpg"))
    if sample_images:
        demo_sample = str(sample_images[0])
        infer_res = run_inference(
            image_path=demo_sample,
            checkpoint_path="outputs/checkpoints/full_rb_unet_latest.pth",
            output_dir=str(demo_dir),
            image_size=128 if smoke_test else 256,
        )
        print(f"Demo artifacts exported to: {demo_dir}")

    total_time = time.time() - start_total_time
    print("\n" + "=" * 75)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f} SECONDS!")
    print(f"All research metrics, checkpoints, plots, and tables are saved in: outputs/")
    print("=" * 75)

    if serve_ui:
        print("\n[UI] Launching web application service...")
        subprocess.run([sys.executable, "app.py"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master RB-UNet One-Command Runner.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Execute rapid dry-run end-to-end")
    parser.add_argument("--serve", action="store_true", help="Launch UI after execution")
    args = parser.parse_args()

    execute_pipeline(config_path=args.config, smoke_test=args.smoke_test, serve_ui=args.serve)
