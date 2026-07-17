#!/usr/bin/env python3
# Frozen CytoFM features -> logistic regression. Sanity check before head tuning.
# python cloud/eval_cytofm_features.py

import sys, numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.models.architectures.cytofm import load_cytofm
from src.data.slide_dataloader import create_data_loader
from src.data.transforms import get_val_transform
from src.data.norm_constants import NORM_CONSTANTS

CKPT = "models/cytofm_weights.pth"
META = "data/metadata/metadata_patient_split.csv"


def extract(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for imgs, y in loader:
            feats.append(model(imgs.to(device)).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_cytofm(CKPT, device)

    tf = get_val_transform(
        mean=NORM_CONSTANTS["IMAGENET_MEAN"],
        std=NORM_CONSTANTS["IMAGENET_STD"],
        resize=(224, 224),
    )

    meta = pd.read_csv(META)
    splits = {}
    for s in ("train", "val", "test"):
        df = meta[meta["split"] == s]
        loader = create_data_loader(df, batch_size=64, shuffle=False, num_workers=4, transform=tf)
        feats, y = extract(backbone, loader, device)
        splits[s] = (feats, y)
        print(f"{s:>5s}: n={len(y)}  classes={np.bincount(y).tolist()}  feat={feats.shape[1]}d")

    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["val"]
    X_te, y_te = splits["test"]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    print(f"{'C':>8s} {'val_acc':>8s} {'val_auc':>8s} {'te_acc':>8s} {'te_auc':>8s}")
    best, best_C = -1, 1.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X_tr, y_tr)

        vauc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        tauc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        va = accuracy_score(y_val, clf.predict(X_val))
        ta = accuracy_score(y_te, clf.predict(X_te))
        print(f"{C:8.2f} {va:8.4f} {vauc:8.4f} {ta:8.4f} {tauc:8.4f}")
        if vauc > best:
            best, best_C = vauc, C

    clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    te_pred = clf.predict(X_te)
    te_prob = clf.predict_proba(X_te)[:, 1]
    print(f"test (C={best_C}): auc={roc_auc_score(y_te, te_prob):.4f} f1={f1_score(y_te, te_pred, average='weighted'):.4f}")
