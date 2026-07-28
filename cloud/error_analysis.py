#!/usr/bin/env python3
# python cloud/error_analysis.py --exp resnet18_aug
# python cloud/error_analysis.py --exp resnet18_aug --dump-errors 20

import argparse, glob, json, shutil, sys
import numpy as np, pandas as pd, torch, yaml
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, ".")
from src.models.load_model import load_model_from_config
from src.data.slide_dataloader import SlideDataset
from src.data.transforms import create_transforms_from_config

METADATA = "data/metadata/metadata_patient_split.csv"
CLASSES = {0: "benign", 1: "malignant"}


def load_model(exp_dir):
    cfg = yaml.safe_load((exp_dir / "configs" / "training_config.yaml").read_text())
    tcfg = json.loads((exp_dir / "configs" / "transform_config.json").read_text())
    _, val_tf = create_transforms_from_config(tcfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_from_config(
        architecture=cfg["architecture"], num_classes=cfg.get("num_classes", 2),
        pretrained=False, model_params=cfg.get("model_params", {}), device=device,
    )
    best = exp_dir / "models" / "best_model.pth"
    w = best if best.exists() else exp_dir / "models" / "final_model.pth"
    model.load_state_dict(torch.load(w, map_location=device, weights_only=True))
    model.eval()
    return model, val_tf, device


def predict_all(model, dataset, device):
    rows = []
    with torch.no_grad():
        for i in range(len(dataset)):
            img, label = dataset[i]
            prob = torch.softmax(model(img.unsqueeze(0).to(device)), 1).cpu().numpy()[0]
            meta = dataset.metadata.iloc[i]
            rows.append(dict(
                filepath=meta["filepath"], label=int(label), pred=int(prob.argmax()),
                conf_malignant=float(prob[1]), patient_id=meta.get("patient_id", ""),
                slide_id=meta.get("slide_id", ""), correct=int(label) == int(prob.argmax()),
            ))
    return pd.DataFrame(rows)


def find_experiment(name):
    dirs = sorted(glob.glob(f"experiments/{name}_*"))
    if not dirs:
        sys.exit(f"no experiment matching '{name}'")
    return Path(dirs[-1])


def print_confusion(df):
    cm = confusion_matrix(df.label, df.pred)
    print(f"confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")


def print_per_patient(df):
    print("per-patient accuracy:")
    stats = df.groupby("patient_id").agg(
        n=("correct", "size"), acc=("correct", "mean"),
        n_ben=("label", lambda x: (x == 0).sum()),
        n_mal=("label", lambda x: (x == 1).sum()),
    ).sort_values("acc")
    for pid, r in stats.iterrows():
        print(f"  {pid}: n={r.n:.0f} ben={r.n_ben:.0f} mal={r.n_mal:.0f} acc={r.acc:.2f}")


def print_worst(df, n=10):
    fp = df[(df.label == 0) & (df.pred == 1)].nlargest(n, "conf_malignant")
    fn = df[(df.label == 1) & (df.pred == 0)].nsmallest(n, "conf_malignant")
    print(f"top {n} false positives:")
    for _, r in fp.iterrows():
        print(f"  conf={r.conf_malignant:.3f} patient={r.patient_id} {Path(r.filepath).name}")
    print(f"top {n} false negatives:")
    for _, r in fn.iterrows():
        print(f"  conf={r.conf_malignant:.3f} patient={r.patient_id} {Path(r.filepath).name}")


def dump_errors(df, out_dir, n=20):
    out = Path(out_dir)
    for sub in ("false_positive", "false_negative"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    fp = df[(df.label == 0) & (df.pred == 1)].nlargest(n, "conf_malignant")
    fn = df[(df.label == 1) & (df.pred == 0)].nsmallest(n, "conf_malignant")

    dataset = SlideDataset(df, transform=None)
    for tag, subset in [("false_positive", fp), ("false_negative", fn)]:
        for _, r in subset.iterrows():
            src = dataset._resolve_path(r.filepath)
            name = f"conf{r.conf_malignant:.3f}_{r.patient_id}_{Path(r.filepath).name}"
            shutil.copy2(src, out / tag / name)
    print(f"copied {len(fp)+len(fn)} error images to {out}/")


def find_optimal_threshold(df):
    """Find threshold on val set that maximizes accuracy."""
    best_t, best_acc = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (df.conf_malignant >= t).astype(int)
        acc = (preds == df.label).mean()
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment name prefix, e.g. resnet18_aug")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--dump-errors", type=int, default=0, help="copy N worst errors to disk")
    args = ap.parse_args()

    exp_dir = find_experiment(args.exp)
    print(f"experiment: {exp_dir.name}")

    meta = pd.read_csv(METADATA)
    split_df = meta[meta["split"] == args.split].copy().reset_index(drop=True)
    print(f"{args.split} set: {len(split_df)} samples")

    model, tf, device = load_model(exp_dir)
    dataset = SlideDataset(split_df, transform=tf)
    df = predict_all(model, dataset, device)

    acc = df.correct.mean()
    print(f"overall accuracy: {acc:.4f}")
    print_confusion(df)
    print_per_patient(df)
    print_worst(df)

    val_df = meta[meta["split"] == "val"].copy().reset_index(drop=True)
    val_dataset = SlideDataset(val_df, transform=tf)
    val_preds = predict_all(model, val_dataset, device)
    opt_t, val_acc = find_optimal_threshold(val_preds)
    test_preds_tuned = (df.conf_malignant >= opt_t).astype(int)
    tuned_acc = (test_preds_tuned == df.label).mean()
    print(f"threshold tuning: optimal={opt_t:.2f} val_acc={val_acc:.4f} test_acc={tuned_acc:.4f} (was {acc:.4f})")

    Path("analysis").mkdir(exist_ok=True)
    if args.dump_errors > 0:
        dump_errors(df, f"analysis/{exp_dir.name}_errors", n=args.dump_errors)

    df.to_csv(f"analysis/{exp_dir.name}_predictions.csv", index=False)
    print(f"predictions saved to analysis/{exp_dir.name}_predictions.csv")
