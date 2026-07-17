#!/usr/bin/env python3
# python cloud/run_kfold.py
# python cloud/run_kfold.py --no-shutdown --folds 3
# python cloud/run_kfold.py --only resnet18_aug

import subprocess, sys, json, glob, argparse
import numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, ".")
from split_by_patient import kfold

META_SRC = "data/metadata/metadata.csv"
FOLD_DIR = Path("data/metadata/folds")

CONFIGS = [
    # Baseline CNN
    ("configs/custom_cnn_no_aug.yaml",        "baseline_cnn_noaug"),
    ("configs/custom_cnn_training.yaml",      "baseline_cnn_aug"),
    # Pretrained CNNs — no color jitter
    ("configs/resnet18_noaug.yaml",           "resnet18_noaug"),
    ("configs/mobilenet_v3_noaug.yaml",       "mobnet_noaug"),
    ("configs/efficientnet_b0_noaug.yaml",    "effnet_noaug"),
    # Pretrained CNNs — with augmentation (color jitter)
    ("configs/resnet18_aug.yaml",             "resnet18_aug_lr1e4"),
    ("configs/resnet18_aug_lr1e3.yaml",       "resnet18_aug_lr1e3"),
    ("configs/efficientnet_b0_aug.yaml",      "effnet_aug_lr1e4"),
    ("configs/efficientnet_b0_aug_lr1e3.yaml","effnet_aug_lr1e3"),
    ("configs/mobilenet_v3_aug.yaml",         "mobnet_aug_lr1e4"),
    ("configs/mobilenet_v3_aug_lr1e3.yaml",   "mobnet_aug_lr1e3"),
    # CytoFM
    ("configs/cytofm_head_only.yaml",         "cytofm_headonly"),
    ("configs/cytofm_finetune.yaml",          "cytofm_finetune"),
]


def make_folds(k, seed):
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(META_SRC).drop(columns=["split", "fold", "patient_id"], errors="ignore")
    df = kfold(df, k=k, seed=seed)
    for i in range(k):
        f = df.copy()
        f["split"] = "train"
        f.loc[f.fold == i, "split"] = "test"
        f.loc[f.fold == (i + 1) % k, "split"] = "val"
        f.to_csv(FOLD_DIR / f"fold_{i}.csv", index=False)
    return k


def run_one(cfg, name, fold):
    return subprocess.run([
        sys.executable, "run_training.py",
        "--config", cfg, "--metadata", str(FOLD_DIR / f"fold_{fold}.csv"),
        "--experiment_name", f"{name}_fold{fold}",
    ]).returncode == 0


def summarize(k):
    for _, name in CONFIGS:
        accs, aucs = [], []
        for i in range(k):
            dirs = sorted(glob.glob(f"experiments/{name}_fold{i}_*"))
            if not dirs:
                continue
            h = Path(dirs[-1]) / "metrics" / "training_history.json"
            if not h.exists():
                continue
            d = json.loads(h.read_text())
            accs.append(d.get("best_val_acc", 0))
            best_auc = d.get("best_val_auc", 0)
            if not best_auc:
                best_auc = max((m.get("auc", 0) for m in d.get("history", {}).get("metrics", [{}])), default=0)
            aucs.append(best_auc)
        if accs:
            print(f"{name:<24s} acc={np.mean(accs):.4f}±{np.std(accs):.4f}  auc={np.mean(aucs):.4f}±{np.std(aucs):.4f}")


def fold_exists(name, fold):
    dirs = sorted(glob.glob(f"experiments/{name}_fold{fold}_*"))
    if not dirs:
        return False
    h = Path(dirs[-1]) / "metrics" / "training_history.json"
    return h.exists()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shutdown", action="store_true")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip folds that already have completed results")
    args = ap.parse_args()

    k = make_folds(args.folds, args.seed)
    cfgs = [(c, n) for c, n in CONFIGS if not args.only or n == args.only]

    for cfg, name in cfgs:
        for i in range(k):
            if args.skip_existing and fold_exists(name, i):
                print(f"{name} fold {i}: skipped (exists)")
                continue
            print(f"{name} fold {i}")
            run_one(cfg, name, i)

    print("results:")
    summarize(k)

    if not args.no_shutdown:
        subprocess.run(["sudo", "shutdown", "-h", "+1"])
