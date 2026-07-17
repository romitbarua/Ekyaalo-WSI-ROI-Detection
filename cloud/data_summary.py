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
    unmatched = df[df["wsi"].isna()]["fname"]
    print(f"filename formats: {matched} matched Z-*.tif, {len(unmatched)} unmatched")
    if len(unmatched) > 0:
        for fn in unmatched.head(5):
            print(f"  {fn}")

    # for unmatched, try splitting on common patterns
    def extract_wsi(fname):
        # try Z-..tif pattern first
        if "Z-" in fname and ".tif" in fname:
            return fname.split("Z-")[1].split(".tif")[0].upper()
        # try just the slide_id-like part before _lux or _x or similar
        parts = fname.replace(".png", "").replace(".jpg", "").split("_lux")[0]
        # if it looks like a slide name, use it
        return parts.upper()

    df["wsi"] = df["fname"].apply(extract_wsi)

    print(f"samples={len(df)} WSIs={df['wsi'].nunique()} patients={df['patient_id'].nunique()}")

    for cls_name, g in df.groupby("class_name"):
        print(f"  {cls_name}: {len(g)} samples, {g['slide_id'].nunique()} slides, {g['patient_id'].nunique()} patients")

    pat = df.groupby("patient_id").agg(
        samples=("label", "size"),
        wsis=("wsi", "nunique"),
        label=("label", lambda x: int(x.mode()[0])),
        cls=("class_name", lambda x: x.mode()[0]),
    ).sort_values(["label", "patient_id"])

    print(f"{'patient':<12s} {'cls':<12s} {'WSIs':>6s} {'samples':>8s}")
    for pid, r in pat.iterrows():
        print(f"{pid:<12s} {r.cls:<12s} {r.wsis:>6d} {r.samples:>8d}")

    if "split" in df.columns:
        print(f"{'split':<8s} {'n':>6s} {'benign':>8s} {'malig':>8s} {'%benign':>8s}")
        for s in ("train", "val", "test"):
            sub = df[df["split"] == s]
            if sub.empty:
                continue
            c = sub["label"].value_counts().sort_index()
            b, m = c.get(0, 0), c.get(1, 0)
            print(f"{s:<8s} {len(sub):>6d} {b:>8d} {m:>8d} {100*b/len(sub):>7.1f}%")
