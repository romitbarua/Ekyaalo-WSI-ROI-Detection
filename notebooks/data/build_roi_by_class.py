"""
Creates notebooks/data/ROI_by_class/{Benign,Malignant,Suspicious,Unmatched}/
with relative symlinks pointing back to data/raw/ROI/.

Class is assigned by matching each ROI file to its counterpart in
AnnotatedDataSet (specimen + lux + luy key), then reading the subdirectory
the padded file lives in. The 77 files with no match go to Unmatched/.

Run from any working directory — all paths are resolved relative to this file.
"""

import re
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
ROI_DIR   = ROOT / "data" / "raw" / "ROI"
ANNOT_DIR = ROOT / "data" / "raw" / "AnnotatedDataSet"
OUT_DIR   = Path(__file__).parent / "ROI_by_class"

CLASSES = ["Benign", "Malignant", "Suspicious", "Unmatched"]

ROI_RE = re.compile(
    r"^(?P<specimen>.+?)_lux_(?P<lux>\d+)_luy_(?P<luy>\d+)_h_(?P<h>\d+)_w_(?P<w>\d+)\.png$"
)
ANNOT_PADDED_RE = re.compile(
    r"^(?P<specimen>.+?)_lux_(?P<lux>\d+)_luy_(?P<luy>\d+)_h_(?P<h>\d+)_w_(?P<w>\d+)_padded_384.*\.png$"
)

def make_key(specimen, lux, luy):
    return f"{specimen}__lux{lux}_luy{luy}"

# Build padded-file key → class lookup
padded_class: dict[str, str] = {}
for cls in ["Benign", "Malignant", "Suspicious"]:
    for f in (ANNOT_DIR / cls).glob("*.png"):
        m = ANNOT_PADDED_RE.match(f.name)
        if m:
            d = m.groupdict()
            k = make_key(d["specimen"], d["lux"], d["luy"])
            padded_class.setdefault(k, cls)  # keep first if duplicate

# Create output directories
for cls in CLASSES:
    (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

# Symlink each ROI file into the right class folder
counts = {cls: 0 for cls in CLASSES}
for roi_file in sorted(ROI_DIR.glob("*.png")):
    m = ROI_RE.match(roi_file.name)
    if not m:
        print(f"[WARN] unparseable: {roi_file.name}")
        continue

    d = m.groupdict()
    k = make_key(d["specimen"], d["lux"], d["luy"])
    cls = padded_class.get(k, "Unmatched")

    link = OUT_DIR / cls / roi_file.name
    if link.exists() or link.is_symlink():
        link.unlink()

    # relative path from the symlink location back to the original
    link.symlink_to(Path("../../../..") / "data" / "raw" / "ROI" / roi_file.name)
    counts[cls] += 1

print(f"ROI_by_class created at: {OUT_DIR.relative_to(ROOT)}")
print()
for cls in CLASSES:
    print(f"  {cls:<12}: {counts[cls]:>4} files")
print(f"  {'TOTAL':<12}: {sum(counts.values()):>4} files")
