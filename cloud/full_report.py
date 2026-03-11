#!/usr/bin/env python3
# python cloud/full_report.py
# Comprehensive model evaluation report for presentation.

import sys, json, glob
import numpy as np, pandas as pd, torch, yaml
from pathlib import Path
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score,
                             classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.models.load_model import load_model_from_config
from src.data.slide_dataloader import create_data_loader
from src.data.transforms import create_transforms_from_config, get_val_transform
from src.data.norm_constants import NORM_CONSTANTS
from src.models.architectures.cytofm import load_cytofm
from split_by_patient import patient_of

META = "data/metadata/metadata_patient_split.csv"
FOLD_DIR = Path("data/metadata/folds")
CLASSES = ["Benign", "Malig/Susp"]

SINGLE_SPLIT_MODELS = [
    ("baseline_cnn_aug",    "Baseline CNN"),
    ("baseline_cnn_no_aug", "Baseline CNN (no aug)"),
    ("resnet18",            "ResNet-18"),
    ("resnet18_aug",        "ResNet-18 + Aug"),
    ("efficientnet_b0",     "EfficientNet-B0"),
    ("efficientnet_b0_aug", "EfficientNet-B0 + Aug"),
    ("mobilenet_v3",        "MobileNet-V3"),
    ("mobilenet_v3_aug",    "MobileNet-V3 + Aug"),
    ("cytofm_head_only",    "CytoFM (head only)"),
    ("cytofm_finetune",     "CytoFM (fine-tune)"),
]

KFOLD_CONFIGS = [
    "baseline_cnn_noaug", "baseline_cnn_aug",
    "resnet18_noaug",     "resnet18_aug_lr1e4", "resnet18_aug_lr1e3",
    "mobnet_noaug",       "mobnet_aug_lr1e4",   "mobnet_aug_lr1e3",
    "effnet_noaug",       "effnet_aug_lr1e4",   "effnet_aug_lr1e3",
    "cytofm_headonly",    "cytofm_finetune",
]

KFOLD_LABELS = {
    "baseline_cnn_noaug": "Baseline CNN",
    "baseline_cnn_aug":   "Baseline CNN + Aug",
    "resnet18_noaug":     "ResNet-18",
    "resnet18_aug_lr1e4": "ResNet-18 Aug (LR 1e-4)",
    "resnet18_aug_lr1e3": "ResNet-18 Aug (LR 1e-3)",
    "effnet_noaug":       "EfficientNet-B0",
    "effnet_aug_lr1e4":   "EffNet-B0 Aug (LR 1e-4)",
    "effnet_aug_lr1e3":   "EffNet-B0 Aug (LR 1e-3)",
    "mobnet_noaug":       "MobileNet-V3",
    "mobnet_aug_lr1e4":   "MobNet-V3 Aug (LR 1e-4)",
    "mobnet_aug_lr1e3":   "MobNet-V3 Aug (LR 1e-3)",
    "cytofm_headonly":    "CytoFM (head-only)",
    "cytofm_finetune":    "CytoFM (fine-tune)",
}

ARCHITECTURES = [
    ("baseline_cnn", "Baseline CNN",    {}),
    ("resnet18",     "ResNet-18",       {}),
    ("efficientnet_b0", "EfficientNet-B0", {}),
    ("mobilenet_v3", "MobileNet-V3",    {}),
    ("cytofm",       "CytoFM",         {"checkpoint_path": "models/cytofm_weights.pth",
                                         "head_hidden_dim": 256, "dropout": 0.3}),
]

CURVE_MODELS = [
    ("resnet18_aug",    "ResNet-18 + Aug"),
    ("mobilenet_v3_aug","MobileNet-V3 + Aug"),
    ("efficientnet_b0", "EfficientNet-B0"),
    ("cytofm_finetune", "CytoFM fine-tune"),
]

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def banner(title):
    print(f"\n{title}")


def load_and_eval(exp_dir, test_df):
    cfg = yaml.safe_load((exp_dir / "configs" / "training_config.yaml").read_text())
    tcfg = json.loads((exp_dir / "configs" / "transform_config.json").read_text())
    _, val_tf = create_transforms_from_config(tcfg)
    loader = create_data_loader(test_df, batch_size=64, shuffle=False, transform=val_tf)

    model = load_model_from_config(
        architecture=cfg["architecture"], num_classes=cfg.get("num_classes", 2),
        pretrained=False, model_params=cfg.get("model_params", {}), device=DEV,
    )
    w = exp_dir / "models" / "best_model.pth"
    if not w.exists():
        w = exp_dir / "models" / "final_model.pth"
    model.load_state_dict(torch.load(w, map_location=DEV, weights_only=True))
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            ps.extend(torch.softmax(model(imgs.to(DEV)), 1).cpu().numpy())
            ys.extend(labels.numpy())
    y, p = np.array(ys), np.array(ps)
    yhat = p.argmax(1)
    return y, yhat, p


def print_metrics(y, yhat, p):
    auc = roc_auc_score(y, p[:, 1])
    acc = accuracy_score(y, yhat)
    f1 = f1_score(y, yhat, average="weighted")
    cm = confusion_matrix(y, yhat)
    print(f"  AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}  n={len(y)}")
    print(f"  Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    return dict(auc=auc, acc=acc, f1=f1)


def find_exp(name):
    dirs = sorted(glob.glob(f"experiments/{name}_[0-9]*"))
    return Path(dirs[-1]) if dirs else None


def section_dataset():
    banner("Dataset Summary")
    meta = pd.read_csv(META)
    meta["patient_id"] = meta["slide_id"].apply(patient_of)

    print(f"  samples={len(meta)} slides={meta['slide_id'].nunique()} patients={meta['patient_id'].nunique()}")
    for s in ("train", "val", "test"):
        sub = meta[meta["split"] == s]
        if sub.empty:
            continue
        b = (sub["label"] == 0).sum()
        m = (sub["label"] == 1).sum()
        print(f"  {s}: n={len(sub)} benign={b} malig={m} ({100*b/len(sub):.1f}% benign)")


def section_complexity():
    banner("Model Complexity")
    x = torch.randn(1, 3, 224, 224).to(DEV)

    for arch, label, params in ARCHITECTURES:
        try:
            m = load_model_from_config(arch, num_classes=2, pretrained=False,
                                       model_params=params, device=DEV).eval()
            n_params = sum(p.numel() for p in m.parameters())
            with FlopCounterMode(display=False) as fc:
                m(x)
            flops = fc.get_total_flops()
            print(f"  {label}: {n_params/1e6:.1f}M params, {flops/1e9:.2f} GFLOPs")
            del m
        except Exception as e:
            print(f"  {label}: ERROR {e}")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def section_single_split():
    banner("Single-Split Test Evaluation")
    meta = pd.read_csv(META)
    test_df = meta[meta["split"] == "test"]
    print(f"  test set: {len(test_df)} samples")

    summary = []
    for name, label in SINGLE_SPLIT_MODELS:
        exp = find_exp(name)
        if not exp:
            continue

        h = json.loads((exp / "metrics" / "training_history.json").read_text())
        val_auc = h.get("best_val_auc", 0)
        val_acc = h.get("best_val_acc", 0)

        try:
            y, yhat, p = load_and_eval(exp, test_df)
            m = print_metrics(y, yhat, p)
            summary.append((label, val_acc, val_auc, m["acc"], m["auc"], m["f1"]))
        except Exception as e:
            print(f"  {label}: error {e}")

    if summary:
        print(f"\n  {'Model':<25s} {'ValAcc':>7s} {'ValAUC':>7s} {'TeAcc':>7s} {'TeAUC':>7s} {'TeF1':>7s}")
        for row in sorted(summary, key=lambda x: -x[4]):
            print(f"  {row[0]:<25s} {row[1]:>7.4f} {row[2]:>7.4f} {row[3]:>7.4f} {row[4]:>7.4f} {row[5]:>7.4f}")


def section_kfold():
    banner("K-Fold Cross-Validation")

    n_folds = 0
    for i in range(20):
        if not (FOLD_DIR / f"fold_{i}.csv").exists():
            break
        n_folds = i + 1
    if n_folds == 0:
        print("  no fold CSVs found")
        return
    print(f"  folds: {n_folds}")

    all_results = {}
    for name in KFOLD_CONFIGS:
        label = KFOLD_LABELS.get(name, name)
        fold_metrics = []

        for i in range(n_folds):
            dirs = sorted(glob.glob(f"experiments/{name}_fold{i}_*"))
            if not dirs:
                continue
            exp = Path(dirs[-1])
            fold_csv = FOLD_DIR / f"fold_{i}.csv"
            if not fold_csv.exists():
                continue

            fold_df = pd.read_csv(fold_csv)
            test_df = fold_df[fold_df["split"] == "test"]
            h = json.loads((exp / "metrics" / "training_history.json").read_text())

            try:
                y, yhat, p = load_and_eval(exp, test_df)
                m = print_metrics(y, yhat, p)
                m["val_auc"] = h.get("best_val_auc", 0)
                m["val_acc"] = h.get("best_val_acc", 0)
                fold_metrics.append(m)
            except Exception as e:
                print(f"  {label} fold {i}: error {e}")

        if fold_metrics:
            all_results[name] = fold_metrics

    if all_results:
        banner("K-Fold Summary")
        print(f"  {'Model':<28s} {'ValAUC':>10s} {'TeAUC':>10s} {'TeAcc':>10s} {'TeF1':>10s}")
        for name in KFOLD_CONFIGS:
            if name not in all_results:
                continue
            fm = all_results[name]
            label = KFOLD_LABELS.get(name, name)
            va = np.mean([m["val_auc"] for m in fm])
            va_s = np.std([m["val_auc"] for m in fm])
            ta = np.mean([m["auc"] for m in fm])
            ta_s = np.std([m["auc"] for m in fm])
            ac = np.mean([m["acc"] for m in fm])
            ac_s = np.std([m["acc"] for m in fm])
            f1 = np.mean([m["f1"] for m in fm])
            f1_s = np.std([m["f1"] for m in fm])
            print(f"  {label:<28s} {va:.4f}±{va_s:.4f} {ta:.4f}±{ta_s:.4f} {ac:.4f}±{ac_s:.4f} {f1:.4f}±{f1_s:.4f}")


def section_curves():
    banner("Training Curves")
    for name, label in CURVE_MODELS:
        exp = find_exp(name)
        if not exp:
            continue
        h = json.loads((exp / "metrics" / "training_history.json").read_text())
        hist = h.get("history", {})
        tl = hist.get("train_loss", [])
        vl = hist.get("val_loss", [])
        metrics = hist.get("metrics", [])
        if not tl:
            continue

        print(f"  {label}:")
        for e in range(len(tl)):
            va_acc = metrics[e].get("accuracy", 0) if e < len(metrics) else 0
            va_auc = metrics[e].get("auc", 0) if e < len(metrics) else 0
            vl_e = vl[e] if e < len(vl) else 0
            print(f"    ep{e+1:02d} tr={tl[e]:.4f} va={vl_e:.4f} acc={va_acc:.4f} auc={va_auc:.4f}")


def section_cytofm_frozen():
    banner("CytoFM Frozen Features (LogReg)")
    ckpt = "models/cytofm_weights.pth"
    if not Path(ckpt).exists():
        print("  CytoFM checkpoint not found")
        return

    backbone = load_cytofm(ckpt, DEV)
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
        feats, labels = [], []
        backbone.eval()
        with torch.no_grad():
            for imgs, y in loader:
                feats.append(backbone(imgs.to(DEV)).cpu().numpy())
                labels.append(y.numpy())
        feats = np.concatenate(feats)
        labels = np.concatenate(labels)
        splits[s] = (feats, labels)
        b = (labels == 0).sum()
        m = (labels == 1).sum()
        print(f"  {s}: n={len(labels)} ({b} benign, {m} malig) {feats.shape[1]}d")

    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["val"]
    X_te, y_te = splits["test"]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    best_score, best_C = -1, 1.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        vauc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        tauc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        va = accuracy_score(y_val, clf.predict(X_val))
        ta = accuracy_score(y_te, clf.predict(X_te))
        print(f"  C={C:.2f} val_acc={va:.4f} val_auc={vauc:.4f} te_acc={ta:.4f} te_auc={tauc:.4f}")
        if vauc > best_score:
            best_score, best_C = vauc, C

    clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    te_pred = clf.predict(X_te)
    te_prob = clf.predict_proba(X_te)[:, 1]

    print(f"  best C={best_C}")
    y, yhat, p = y_te, te_pred, np.column_stack([1 - te_prob, te_prob])
    print_metrics(y, yhat, p)

    del backbone
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    print("WSI ROI Detection - Model Evaluation Report")
    section_dataset()
    section_complexity()
    section_single_split()
    section_kfold()
    section_curves()
    section_cytofm_frozen()
    print("\ndone")
