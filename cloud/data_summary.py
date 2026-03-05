#!/usr/bin/env python3
# python cloud/data_summary.py
# python cloud/data_summary.py --metadata data/metadata/metadata_patient_split.csv

import argparse, sys
import pandas as pd, numpy as np

sys.path.insert(0, ".")
from split_by_patient import patient_of

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata/metadata.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.metadata)
    df["patient_id"] = df["slide_id"].apply(patient_of)

    # extract actual WSI name: the part between "Z-" and ".tif" in the filename
    # e.g. "2025-04-16T21-36-24-411Z-C1432-23-1.tif_lux_..." -> "C1432-23-1"
    df["wsi"] = df["fname"].str.extract(r'Z-(.+?)\.tif', expand=False).str.upper()
    matched = df["wsi"].notna().sum()
    df.loc[df["wsi"].isna(), "wsi"] = df.loc[df["wsi"].isna(), "slide_id"]

    print(f"samples:  {len(df)}")
    print(f"WSIs:     {df['wsi'].nunique()}  (regex matched {matched}/{len(df)})")
    print(f"patients: {df['patient_id'].nunique()}")
    # show a few extracted values
    print(f"\nWSI sample: {df['wsi'].unique()[:5].tolist()}")

    # per-class
    for cls_name, g in df.groupby("class_name"):
        print(f"\n  {cls_name}: {len(g)} samples, {g['slide_id'].nunique()} slides, {g['patient_id'].nunique()} patients")

    # per-patient breakdown
    pat = df.groupby("patient_id").agg(
        samples=("label", "size"),
        wsis=("wsi", "nunique"),
        label=("label", lambda x: int(x.mode()[0])),
        cls=("class_name", lambda x: x.mode()[0]),
    ).sort_values(["label", "patient_id"])

    print(f"\n{'patient':<12s} {'cls':<12s} {'WSIs':>6s} {'samples':>8s}")
    print("-" * 40)
    for pid, r in pat.iterrows():
        print(f"{pid:<12s} {r.cls:<12s} {r.wsis:>6d} {r.samples:>8d}")

    # split distribution (if present)
    if "split" in df.columns:
        print(f"\n{'split':<8s} {'n':>6s} {'benign':>8s} {'malig':>8s} {'%benign':>8s}")
        print("-" * 42)
        for s in ("train", "val", "test"):
            sub = df[df["split"] == s]
            if sub.empty:
                continue
            c = sub["label"].value_counts().sort_index()
            b, m = c.get(0, 0), c.get(1, 0)
            print(f"{s:<8s} {len(sub):>6d} {b:>8d} {m:>8d} {100*b/len(sub):>7.1f}%")
