#!/usr/bin/env python3
# python cloud/eval_kfold.py
# python cloud/eval_kfold.py --folds 3

import argparse, glob, json, sys
import numpy as np, pandas as pd, torch, yaml
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, ".")
from src.models.load_model import load_model_from_config
from src.data.slide_dataloader import create_data_loader
from src.data.transforms import create_transforms_from_config

FOLD_DIR = Path("data/metadata/folds")

CONFIGS = [
    "resnet18_aug_lr1e4",
    "resnet18_aug_lr1e3",
    "effnet_aug_lr1e4",
    "effnet_aug_lr1e3",
    "mobnet_aug_lr1e4",
    "mobnet_aug_lr1e3",
]


def eval_one(exp_dir, test_df):
    cfg = yaml.safe_load((exp_dir / "configs" / "training_config.yaml").read_text())
    tcfg = json.loads((exp_dir / "configs" / "transform_config.json").read_text())
    _, tf = create_transforms_from_config(tcfg)
    loader = create_data_loader(test_df, batch_size=64, shuffle=False, transform=tf)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_from_config(
        architecture=cfg["architecture"], num_classes=cfg.get("num_classes", 2),
        pretrained=False, model_params=cfg.get("model_params", {}), device=device,
    )
    w = exp_dir / "models" / "best_model.pth"
    if not w.exists():
        w = exp_dir / "models" / "final_model.pth"
    model.load_state_dict(torch.load(w, map_location=device, weights_only=True))
    model.eval()

    y_all, p_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            p_all.extend(torch.softmax(model(imgs.to(device)), 1).cpu().numpy())
            y_all.extend(labels.numpy())
    y, p = np.array(y_all), np.array(p_all)
    yhat = p.argmax(1)
    return dict(acc=accuracy_score(y, yhat), auc=roc_auc_score(y, p[:, 1]),
                f1=f1_score(y, yhat, average="weighted"), n=len(y))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    print(f"{'model':<24s} {'acc':>14s} {'auc':>14s} {'f1':>14s}")

    for name in CONFIGS:
        accs, aucs, f1s = [], [], []
        for i in range(args.folds):
            dirs = sorted(glob.glob(f"experiments/{name}_fold{i}_*"))
            fold_meta = FOLD_DIR / f"fold_{i}.csv"
            if not dirs or not fold_meta.exists():
                continue
            test_df = pd.read_csv(fold_meta)
            test_df = test_df[test_df["split"] == "test"].copy()
            m = eval_one(Path(dirs[-1]), test_df)
            accs.append(m["acc"]); aucs.append(m["auc"]); f1s.append(m["f1"])

        if accs:
            print(f"{name:<24s} {np.mean(accs):.4f}±{np.std(accs):.4f}  {np.mean(aucs):.4f}±{np.std(aucs):.4f}  {np.mean(f1s):.4f}±{np.std(f1s):.4f}")
