#!/usr/bin/env python3
# python generate_metadata.py
# python generate_metadata.py --data-root data/raw/AnnotatedDataSet -o data/metadata/metadata.csv

import argparse
from pathlib import Path
import pandas as pd
from PIL import Image

EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
#LABELS = {"Benign": 0, "Malignant": 1, "Suspicious": 1}
LABELS = {"Benign": 0, "Malignant": 1}

def sid(name):
    # "2025-04-16T21-36-24-411Z-C1432-23-1.tif_lux_..." -> "C1432-23-1"
    if "Z-" in name and ".tif" in name:
        try: return name.split("Z-")[1].split(".tif")[0]
        except Exception: pass
    # "C016-23-3_lux_13115_luy_22733_h_106_w_106_padded_384.png" -> "C016-23-3"
    if "_lux_" in name:
        return name.split("_lux_")[0]
    return name

def scan(root):
    root = Path(root)
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name not in LABELS:
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in EXT:
                continue
            w, h = Image.open(f).size
            rows.append(dict(filepath="/" + str(f), fname=f.name,
                             class_name=d.name, label=LABELS[d.name],
                             slide_id=sid(f.name), width=w, height=h))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/raw/AnnotatedDataSet")
    p.add_argument("--output", "-o", default="data/metadata/metadata.csv")
    a = p.parse_args()

    df = scan(a.data_root)
    df.to_csv(a.output, index=False)
    print(f"{len(df)} images  {df.class_name.value_counts().to_dict()}")
