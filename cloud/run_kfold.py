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
    ("configs/resnet18_aug.yaml",        "resnet18_aug"),
    ("configs/efficientnet_b0_aug.yaml", "efficientnet_b0_aug"),
    ("configs/mobilenet_v3_aug.yaml",    "mobilenet_v3_aug"),
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
            aucs.append(d["history"]["metrics"][-1].get("auc", 0))
        if accs:
            print(f"{name:<24s} acc={np.mean(accs):.4f}±{np.std(accs):.4f}  auc={np.mean(aucs):.4f}±{np.std(aucs):.4f}  [{', '.join(f'{a:.3f}' for a in aucs)}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shutdown", action="store_true")
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    k = make_folds(args.folds, args.seed)
    cfgs = [(c, n) for c, n in CONFIGS if not args.only or n == args.only]

    for cfg, name in cfgs:
        for i in range(k):
            print(f"\n>>> {name} fold {i}")
            run_one(cfg, name, i)

    print("\n--- kfold results ---")
    summarize(k)

    if not args.no_shutdown:
        subprocess.run(["sudo", "shutdown", "-h", "+1"])
