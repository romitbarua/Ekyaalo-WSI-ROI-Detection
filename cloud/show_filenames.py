#!/usr/bin/env python3
import pandas as pd
df = pd.read_csv("data/metadata/metadata_patient_split.csv")
for cls in df["class_name"].unique():
    print(f"{cls}:")
    for fn in df[df["class_name"] == cls]["fname"].head(10):
        print(f"  {fn}")
tifs = df["fname"].str.extract(r'(.*\.tif)', expand=False).dropna().unique()
print(f"unique .tif stems: {len(tifs)}")
for t in sorted(tifs)[:20]:
    print(f"  {t}")
if len(tifs) > 20:
    print(f"  ...and {len(tifs)-20} more")
