#!/usr/bin/env python3
# python cloud/run_all.py
# python cloud/run_all.py --no-shutdown
# python cloud/run_all.py --only resnet18

import subprocess, sys, json, glob, argparse
from pathlib import Path

METADATA = "data/metadata/metadata_patient_split.csv"

EXPERIMENTS = [
    ("configs/custom_cnn_training.yaml",       "baseline_cnn_aug"),
    ("configs/custom_cnn_no_aug.yaml",         "baseline_cnn_no_aug"),
    ("configs/training_example.yaml",          "resnet18"),
    ("configs/resnet18_aug.yaml",              "resnet18_aug"),
    ("configs/efficientnet_b0_training.yaml",  "efficientnet_b0"),
    ("configs/efficientnet_b0_aug.yaml",       "efficientnet_b0_aug"),
    ("configs/mobilenet_v3_training.yaml",     "mobilenet_v3"),
    ("configs/mobilenet_v3_aug.yaml",          "mobilenet_v3_aug"),
    ("configs/cytofm_head_only.yaml",          "cytofm_head_only"),
    ("configs/cytofm_finetune.yaml",           "cytofm_finetune"),
]

def run(config, name):
    if not Path(config).exists():
        print(f"skip {name}: {config} missing")
        return False
    print(f"running {name}")
    rc = subprocess.run([
        sys.executable, "run_training.py",
        "--config", config, "--metadata", METADATA, "--experiment_name", name,
    ]).returncode
    return rc == 0

def results():
    for _, name in EXPERIMENTS:
        dirs = sorted(glob.glob(f"experiments/{name}_[0-9]*"))
        if not dirs:
            continue
        h = Path(dirs[-1]) / "metrics" / "training_history.json"
        if not h.exists():
            continue
        d = json.loads(h.read_text())
        best_auc = d.get("best_val_auc", 0)
        if not best_auc:
            best_auc = max((m.get("auc", 0) for m in d.get("history", {}).get("metrics", [{}])), default=0)
        print(f"{name:20s}  acc={d.get('best_val_acc',0):.4f}  auc={best_auc:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-shutdown", action="store_true")
    p.add_argument("--only", type=str, default=None)
    args = p.parse_args()

    assert Path(METADATA).exists(), f"{METADATA} not found -- run setup.sh first"

    exps = [(c, n) for c, n in EXPERIMENTS if not args.only or n == args.only]
    failed = [n for c, n in exps if not run(c, n)]

    print("results:")
    results()
    if failed:
        print(f"failed: {', '.join(failed)}")

    if not args.no_shutdown:
        subprocess.run(["sudo", "shutdown", "-h", "+1"])
