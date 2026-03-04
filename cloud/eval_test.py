#!/usr/bin/env python3
# Evaluate saved models on the held-out test set.
# Run alongside training -- only needs the final_model.pth files.
#
#   python cloud/eval_test.py
#   python cloud/eval_test.py --only baseline_cnn

import argparse, glob, json, sys
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

sys.path.insert(0, ".")
from src.models.load_model import load_model_from_config
from src.data.slide_dataloader import create_data_loader
from src.data.transforms import create_transforms_from_config, load_transform_config

METADATA = "data/metadata/metadata_patient_split.csv"

def find_experiments():
    out = {}
    for d in sorted(glob.glob("experiments/*")):
        p = Path(d)
        cfg = p / "configs" / "training_config.yaml"
        weights = p / "models" / "final_model.pth"
        if cfg.exists() and weights.exists():
            name = p.name.rsplit("_", 2)[0]
            out[name] = p
    return out

def evaluate(exp_dir, test_df):
    cfg = yaml.safe_load((exp_dir / "configs" / "training_config.yaml").read_text())
    tcfg = json.loads((exp_dir / "configs" / "transform_config.json").read_text())

    _, val_transform = create_transforms_from_config(tcfg)
    loader = create_data_loader(test_df, batch_size=64, shuffle=False, transform=val_transform)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_from_config(
        architecture=cfg["architecture"],
        num_classes=cfg.get("num_classes", 2),
        pretrained=False,
        model_params=cfg.get("model_params", {}),
        device=device,
    )
    model.load_state_dict(torch.load(exp_dir / "models" / "final_model.pth", map_location=device))
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            probs = torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    y = np.array(all_labels)
    p = np.array(all_probs)
    yhat = p.argmax(1)

    return {
        "acc": accuracy_score(y, yhat),
        "f1": f1_score(y, yhat, average="weighted"),
        "auc": roc_auc_score(y, p[:, 1]),
        "prec": precision_score(y, yhat, average="weighted", zero_division=0),
        "recall": recall_score(y, yhat, average="weighted", zero_division=0),
        "n": len(y),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    meta = pd.read_csv(METADATA)
    test_df = meta[meta["split"] == "test"].copy()
    print(f"test set: {len(test_df)} samples\n")

    exps = find_experiments()
    if args.only:
        exps = {k: v for k, v in exps.items() if k == args.only}

    print(f"{'model':<22s} {'acc':>6s} {'f1':>6s} {'auc':>6s} {'prec':>6s} {'rec':>6s}")
    print("-" * 58)
    for name, exp_dir in exps.items():
        try:
            m = evaluate(exp_dir, test_df)
            print(f"{name:<22s} {m['acc']:6.4f} {m['f1']:6.4f} {m['auc']:6.4f} {m['prec']:6.4f} {m['recall']:6.4f}")
        except Exception as e:
            print(f"{name:<22s} ERROR: {e}")
