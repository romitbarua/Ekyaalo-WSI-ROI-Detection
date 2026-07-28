#!/usr/bin/env python3
"""K-fold cross-validation for CytoFM frozen features + LogReg."""

import sys, json, argparse
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, ".")
from src.models.architectures.cytofm import load_cytofm
from src.data.slide_dataloader import create_data_loader
from src.data.transforms import get_val_transform
from src.data.norm_constants import NORM_CONSTANTS
from split_by_patient import kfold

META_SRC = "data/metadata/metadata.csv"
FOLD_DIR = Path("data/metadata/folds")
CKPT = "models/cytofm_weights.pth"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def extract_features(backbone, df, tf):
    loader = create_data_loader(df, batch_size=64, shuffle=False, num_workers=4, transform=tf)
    feats, labels = [], []
    backbone.eval()
    with torch.no_grad():
        for imgs, y in loader:
            feats.append(backbone(imgs.to(DEV)).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not Path(CKPT).exists():
        print(f"ERROR: CytoFM checkpoint not found at {CKPT}")
        sys.exit(1)

    k = make_folds(args.folds, args.seed)

    backbone = load_cytofm(CKPT, torch.device(DEV))
    tf = get_val_transform(
        mean=NORM_CONSTANTS["IMAGENET_MEAN"],
        std=NORM_CONSTANTS["IMAGENET_STD"],
        resize=(224, 224),
    )

    fold_results = []
    for i in range(k):
        fold_csv = FOLD_DIR / f"fold_{i}.csv"
        df = pd.read_csv(fold_csv)

        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]

        print(f"fold {i}: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

        X_tr, y_tr = extract_features(backbone, train_df, tf)
        X_val, y_val = extract_features(backbone, val_df, tf)
        X_te, y_te = extract_features(backbone, test_df, tf)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        X_te = scaler.transform(X_te)

        best_vauc, best_C = -1, 1.0
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            clf.fit(X_tr, y_tr)
            vauc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
            if vauc > best_vauc:
                best_vauc, best_C = vauc, C

        clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")
        clf.fit(X_tr, y_tr)

        te_prob = clf.predict_proba(X_te)[:, 1]
        te_pred = clf.predict(X_te)

        tauc = roc_auc_score(y_te, te_prob)
        tacc = accuracy_score(y_te, te_pred)
        tf1 = f1_score(y_te, te_pred, average="weighted")

        print(f"  Best C={best_C:.2f}  val_auc={best_vauc:.4f}")
        print(f"  Test: AUC={tauc:.4f}  Acc={tacc:.4f}  F1={tf1:.4f}")

        fold_results.append(dict(fold=i, best_C=best_C, val_auc=best_vauc,
                                 test_auc=tauc, test_acc=tacc, test_f1=tf1))

    del backbone
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"CytoFM frozen + LogReg {k}-fold:")
    aucs = [r["test_auc"] for r in fold_results]
    accs = [r["test_acc"] for r in fold_results]
    f1s = [r["test_f1"] for r in fold_results]
    for r in fold_results:
        print(f"  fold {r['fold']}: auc={r['test_auc']:.4f} acc={r['test_acc']:.4f} f1={r['test_f1']:.4f}")
    print(f"  mean: auc={np.mean(aucs):.4f}±{np.std(aucs):.4f} "
          f"acc={np.mean(accs):.4f}±{np.std(accs):.4f} "
          f"f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    out = Path("experiments/cytofm_frozen_kfold_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fold_results, indent=2))
    print(f"saved to {out}")
