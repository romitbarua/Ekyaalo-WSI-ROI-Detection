#!/usr/bin/env python3
# python cloud/show_filenames.py
import pandas as pd
df = pd.read_csv("data/metadata/metadata_patient_split.csv")
# show 10 filenames per class
for cls in df["class_name"].unique():
    print(f"\n=== {cls} ===")
    for fn in df[df["class_name"] == cls]["fname"].head(10):
        print(f"  {fn}")
# unique .tif names (everything before _lux)
tifs = df["fname"].str.extract(r'(.*\.tif)', expand=False).dropna().unique()
print(f"\nunique .tif stems: {len(tifs)}")
for t in sorted(tifs)[:20]:
    print(f"  {t}")
if len(tifs) > 20:
    print(f"  ... and {len(tifs)-20} more")
