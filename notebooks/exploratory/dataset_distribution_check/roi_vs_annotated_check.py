"""
Check whether AnnotatedDataSet is a padded version of ROI.

Questions answered:
  1. Do they share the same files? (matched on specimen + lux + luy)
  2. Are ROI pixel dimensions smaller than AnnotatedDataSet pixel dimensions?
  3. Do the h/w values embedded in filenames match actual pixel dimensions?
"""

import re
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
ROI_DIR    = ROOT / "data" / "raw" / "ROI"
ANNOT_DIR  = ROOT / "data" / "raw" / "AnnotatedDataSet"
CLASSES    = ["Benign", "Malignant", "Suspicious"]

# ---------------------------------------------------------------------------
# filename parsers
# ---------------------------------------------------------------------------

ROI_RE = re.compile(
    r"^(?P<specimen>.+?)_lux_(?P<lux>\d+)_luy_(?P<luy>\d+)_h_(?P<h>\d+)_w_(?P<w>\d+)\.png$"
)
# Two formats exist in AnnotatedDataSet:
#   1. timestamp-based: {ts}-{specimen}.tif_lux_...h_w.png
#   2. padded:          {specimen}_lux_...h_w_padded_384[...].png  ← corresponds to ROI
ANNOT_TIMESTAMP_RE = re.compile(
    r"^.+?-(?P<specimen>C\d+[-\w]+?)\.tif_lux_(?P<lux>\d+)_luy_(?P<luy>\d+)_h_(?P<h>\d+)_w_(?P<w>\d+)\.png$"
)
ANNOT_PADDED_RE = re.compile(
    r"^(?P<specimen>.+?)_lux_(?P<lux>\d+)_luy_(?P<luy>\d+)_h_(?P<h>\d+)_w_(?P<w>\d+)_padded_384.*\.png$"
)

def parse_roi(name: str) -> dict | None:
    m = ROI_RE.match(name)
    return m.groupdict() if m else None

def parse_annot(name: str) -> tuple[str, dict] | tuple[None, None]:
    """Returns (format_type, parsed_dict) or (None, None)."""
    m = ANNOT_PADDED_RE.match(name)
    if m:
        return "padded", m.groupdict()
    m = ANNOT_TIMESTAMP_RE.match(name)
    if m:
        return "timestamp", m.groupdict()
    return None, None

def key(d: dict) -> str:
    return f"{d['specimen']}__lux{d['lux']}_luy{d['luy']}"

# ---------------------------------------------------------------------------
# build lookup tables
# ---------------------------------------------------------------------------

roi_by_key: dict[str, Path] = {}
for f in ROI_DIR.glob("*.png"):
    p = parse_roi(f.name)
    if p:
        roi_by_key[key(p)] = f
    else:
        print(f"  [WARN] unparseable ROI file: {f.name}")

annot_by_key: dict[str, tuple[str, str, Path]] = {}  # key -> (class, fmt, path)
annot_fmt_counts: dict[str, int] = defaultdict(int)
annot_unparseable = 0
for cls in CLASSES:
    for f in (ANNOT_DIR / cls).glob("*.png"):
        fmt, p = parse_annot(f.name)
        if p:
            k = key(p)
            annot_fmt_counts[fmt] += 1
            if k in annot_by_key:
                pass  # duplicates exist; keep first seen
            else:
                annot_by_key[k] = (cls, fmt, f)
        else:
            annot_unparseable += 1

print(f"  AnnotatedDataSet format breakdown:")
for fmt, cnt in sorted(annot_fmt_counts.items()):
    print(f"    {fmt:12s}: {cnt}")
if annot_unparseable:
    print(f"    unparseable  : {annot_unparseable}")

roi_keys   = set(roi_by_key)
annot_keys = set(annot_by_key)

shared    = roi_keys & annot_keys
roi_only  = roi_keys - annot_keys   # in ROI but missing from AnnotatedDataSet
annot_only = annot_keys - roi_keys  # in AnnotatedDataSet but not in ROI

# ---------------------------------------------------------------------------
# report 1: file overlap
# ---------------------------------------------------------------------------

print("=" * 64)
print("FILE OVERLAP  (matched on specimen + lux + luy)")
print("=" * 64)
print(f"  ROI total             : {len(roi_keys):>5}")
print(f"  AnnotatedDataSet total: {len(annot_keys):>5}")
print(f"  Shared                : {len(shared):>5}  ({100*len(shared)/len(roi_keys):.1f}% of ROI)")
print(f"  ROI-only (not in Ann.): {len(roi_only):>5}")
print(f"  Ann.-only (not in ROI): {len(annot_only):>5}")

if roi_only:
    print(f"\n  Sample ROI-only keys (first 5):")
    for k in sorted(roi_only)[:5]:
        print(f"    {k}")

if annot_only:
    print(f"\n  Sample Ann.-only keys (first 5):")
    for k in sorted(annot_only)[:5]:
        print(f"    {k}  [{annot_by_key[k][0]}]")

# ---------------------------------------------------------------------------
# report 2: filename-embedded h/w vs actual pixel dimensions
# ---------------------------------------------------------------------------

print()
print("=" * 64)
print("FILENAME h/w  vs  ACTUAL PIXEL SIZE  (sample of 40 matched pairs)")
print("=" * 64)

sample_keys = random.sample(sorted(shared), min(40, len(shared)))

roi_larger  = 0   # ROI pixel area > annotated pixel area
same_size   = 0
annot_larger = 0

roi_fn_matches    = 0  # filename h/w == actual pixel size for ROI
annot_fn_matches  = 0  # filename h/w == actual pixel size for annotated

print(f"\n{'Key (specimen + coords)':<45} {'ROI px (HxW)':>12} {'Ann px (HxW)':>12}  {'ROI fn (HxW)':>12} {'Ann fn (HxW)':>12}")
print("-" * 100)

dim_data = []
for k in sample_keys:
    roi_path = roi_by_key[k]
    _, _, annot_path = annot_by_key[k]

    roi_img   = Image.open(roi_path)
    annot_img = Image.open(annot_path)

    roi_actual_w,   roi_actual_h   = roi_img.size    # PIL gives (W, H)
    annot_actual_w, annot_actual_h = annot_img.size

    roi_parse        = parse_roi(roi_path.name)
    _, annot_parsed  = parse_annot(annot_path.name)

    roi_fn_h, roi_fn_w     = int(roi_parse["h"]),    int(roi_parse["w"])
    annot_fn_h, annot_fn_w = int(annot_parsed["h"]), int(annot_parsed["w"])

    roi_fn_match   = (roi_fn_h == roi_actual_h   and roi_fn_w == roi_actual_w)
    annot_fn_match = (annot_fn_h == annot_actual_h and annot_fn_w == annot_actual_w)

    if roi_fn_match:   roi_fn_matches   += 1
    if annot_fn_match: annot_fn_matches += 1

    roi_area   = roi_actual_h * roi_actual_w
    annot_area = annot_actual_h * annot_actual_w
    if roi_area < annot_area:
        roi_larger += 0; annot_larger += 1
    elif roi_area > annot_area:
        roi_larger += 1
    else:
        same_size += 1

    label = k.split("__")[0][:20] + "  " + k.split("__")[1][:20]
    print(
        f"{label:<45}"
        f"  {roi_actual_h}x{roi_actual_w:>5}"
        f"  {annot_actual_h}x{annot_actual_w:>5}"
        f"  {'✓' if roi_fn_match else '✗'} {roi_fn_h}x{roi_fn_w:>5}"
        f"  {'✓' if annot_fn_match else '✗'} {annot_fn_h}x{annot_fn_w:>5}"
    )
    dim_data.append((roi_actual_h, roi_actual_w, annot_actual_h, annot_actual_w))

print()
print("=" * 64)
print("DIMENSION SUMMARY  (over sampled pairs)")
print("=" * 64)
print(f"  ROI pixel area > AnnotatedDataSet : {roi_larger}")
print(f"  Same pixel area                   : {same_size}")
print(f"  Ann. pixel area > ROI             : {annot_larger}")
print()
print(f"  ROI filename h/w matches actual pixels     : {roi_fn_matches}/{len(sample_keys)}")
print(f"  Annot filename h/w matches actual pixels   : {annot_fn_matches}/{len(sample_keys)}")

roi_hs   = [d[0] for d in dim_data]
roi_ws   = [d[1] for d in dim_data]
ann_hs   = [d[2] for d in dim_data]
ann_ws   = [d[3] for d in dim_data]
print()
print(f"  ROI   actual h: min={min(roi_hs)}, max={max(roi_hs)}, mean={sum(roi_hs)/len(roi_hs):.1f}")
print(f"  ROI   actual w: min={min(roi_ws)}, max={max(roi_ws)}, mean={sum(roi_ws)/len(roi_ws):.1f}")
print(f"  Annot actual h: min={min(ann_hs)}, max={max(ann_hs)}, mean={sum(ann_hs)/len(ann_hs):.1f}")
print(f"  Annot actual w: min={min(ann_ws)}, max={max(ann_ws)}, mean={sum(ann_ws)/len(ann_ws):.1f}")

unique_annot_sizes = set(zip(ann_hs, ann_ws))
print()
print(f"  Unique AnnotatedDataSet pixel sizes in sample: {sorted(unique_annot_sizes)}")
