import pandas as pd
from pathlib import Path
import shutil

meta = pd.read_csv("data/metadata/fnac_metadata.csv")
test_rows = meta[meta["split"] == "test"]

out_dir = Path("data/test_images")
out_dir.mkdir(parents=True, exist_ok=True)

for _, row in test_rows.iterrows():
    src = Path(row["filepath"])
    shutil.copy2(src, out_dir / src.name)

print(f"Copied {len(test_rows)} test images to {out_dir}")