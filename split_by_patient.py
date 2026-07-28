# Re-split metadata.csv at the patient level (not slide level) to prevent data leakage.
#
# The original split let slides from the same patient (e.g. C016-23-1 and C016-23-2)
# end up in different splits. This script groups by patient ID first, so all of a
# patient's slides stay together.
#
# Patient ID = first two tokens of slide_id:  C016-23-1 → C016-23
#
# python split_by_patient.py
# python split_by_patient.py --val 0.2 --test 0.15 --seed 123
# python split_by_patient.py --kfold 13  # leave-one-patient-out
# python split_by_patient.py --output my_split.csv

import pandas as pd, numpy as np, argparse

def patient_of(sid):
    p = sid.strip().upper().split("-")
    return f"{p[0]}-{p[1]}" if len(p) >= 2 else sid.upper()

def _majority_label(df, patient_id):
    """Per-patient majority class for stratification."""
    return df.groupby(patient_id)["label"].agg(lambda x: int(x.mode()[0]))

def split(df, val=0.15, test=0.15, seed=42):
    df = df.copy()
    df["patient_id"] = df["slide_id"].apply(patient_of)

    # stratify by per-patient majority label so class ratios stay consistent
    pat_label = _majority_label(df, "patient_id")
    rng = np.random.RandomState(seed)

    labels = {}
    for cls in pat_label.unique():
        pats = pat_label[pat_label == cls].index.values
        pats = rng.permutation(pats)
        n_te = max(1, round(len(pats) * test))
        n_va = max(1, round(len(pats) * val))
        for p in pats[:n_te]:
            labels[p] = "test"
        for p in pats[n_te:n_te + n_va]:
            labels[p] = "val"
        for p in pats[n_te + n_va:]:
            labels[p] = "train"

    df["split"] = df["patient_id"].map(labels)
    return df

def kfold(df, k=5, seed=42):
    df = df.copy()
    df["patient_id"] = df["slide_id"].apply(patient_of)

    pat_label = _majority_label(df, "patient_id")
    rng = np.random.RandomState(seed)

    fold_map = {}
    for cls in pat_label.unique():
        pats = pat_label[pat_label == cls].index.values
        pats = rng.permutation(pats)
        for i, p in enumerate(pats):
            fold_map[p] = i % k

    df["fold"] = df["patient_id"].map(fold_map)
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/metadata/metadata.csv")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kfold", type=int, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df = df.drop(columns=["split", "fold", "patient_id"], errors="ignore")

    if args.kfold:
        df = kfold(df, k=args.kfold, seed=args.seed)
    else:
        df = split(df, val=args.val, test=args.test, seed=args.seed)

    dst = args.output or args.input.replace(".csv", "_patient_split.csv")
    df.to_csv(dst, index=False)
    print(dst)

    if "split" in df.columns:
        for s in ("train", "val", "test"):
            sub = df[df["split"] == s]
            counts = sub["label"].value_counts().sort_index().to_dict()
            print(f"  {s:>5s}: n={len(sub)}  labels={counts}")
