"""
For each class, sample 2 matched pairs (ROI original vs padded 384x384)
and save a side-by-side comparison PNG.

Output: benign_pairs.png, malignant_pairs.png, suspicious_pairs.png
Layout per PNG (4 images):

  ┌─────────────────────────────────────────────────────────┐
  │         Pair 1              │         Pair 2             │
  │  [ROI original]             │  [ROI original]            │
  │  (variable size)            │  (variable size)           │
  │                             │                            │
  │  [Padded 384×384]           │  [Padded 384×384]          │
  └─────────────────────────────────────────────────────────┘
"""

import re
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

SEED = 42
N_PAIRS = 2
CLASSES = ["Benign", "Malignant", "Suspicious"]

ROOT      = Path(__file__).resolve().parents[3]
ROI_DIR   = ROOT / "data" / "raw" / "ROI"
ANNOT_DIR = ROOT / "data" / "raw" / "AnnotatedDataSet"
OUT_DIR   = Path(__file__).parent

random.seed(SEED)

# ---------------------------------------------------------------------------
# filename parsers (same logic as roi_vs_annotated_check.py)
# ---------------------------------------------------------------------------

ROI_RE = re.compile(
    r"^(?P<specimen>.+?)_lux_(?P<lux>\d+)_luy_(?P<luy>\d+)_h_(?P<h>\d+)_w_(?P<w>\d+)\.png$"
)
ANNOT_PADDED_RE = re.compile(
    r"^(?P<specimen>.+?)_lux_(?P<lux>\d+)_luy_(?P<luy>\d+)_h_(?P<h>\d+)_w_(?P<w>\d+)_padded_384.*\.png$"
)

def roi_key(name):
    m = ROI_RE.match(name)
    if not m: return None, None
    d = m.groupdict()
    return f"{d['specimen']}__lux{d['lux']}_luy{d['luy']}", d

def padded_key(name):
    m = ANNOT_PADDED_RE.match(name)
    if not m: return None, None
    d = m.groupdict()
    return f"{d['specimen']}__lux{d['lux']}_luy{d['luy']}", d

# ---------------------------------------------------------------------------
# build lookup: key -> (roi_path, padded_path, class, parsed)
# ---------------------------------------------------------------------------

roi_map = {}
for f in ROI_DIR.glob("*.png"):
    k, d = roi_key(f.name)
    if k:
        roi_map[k] = (f, d)

padded_map = {}  # key -> (class, path, parsed)
for cls in CLASSES:
    for f in (ANNOT_DIR / cls).glob("*.png"):
        k, d = padded_key(f.name)
        if k and k not in padded_map:
            padded_map[k] = (cls, f, d)

shared_by_class = {cls: [] for cls in CLASSES}
for k in set(roi_map) & set(padded_map):
    cls = padded_map[k][0]
    shared_by_class[cls].append(k)

# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def load(path): return Image.open(path).convert("RGB")

for cls in CLASSES:
    keys = random.sample(sorted(shared_by_class[cls]), N_PAIRS)

    # 2 rows × N_PAIRS cols; top row = ROI, bottom row = padded
    fig, axes = plt.subplots(2, N_PAIRS, figsize=(5 * N_PAIRS, 10))
    fig.patch.set_facecolor("#1a1a1a")

    fig.suptitle(
        f"{cls}  —  ROI original  vs  Padded 384×384",
        fontsize=14, color="white", fontweight="bold", y=1.01,
    )

    row_labels = ["ROI\n(original)", "Padded\n(384×384)"]
    label_colors = ["#e07b54", "#5b9bd5"]

    for col, k in enumerate(keys):
        roi_path   = roi_map[k][0]
        roi_d      = roi_map[k][1]
        _, pad_path, pad_d = padded_map[k]

        roi_img = load(roi_path)
        pad_img = load(pad_path)

        roi_w, roi_h = roi_img.size
        pad_w, pad_h = pad_img.size

        specimen = roi_d["specimen"]
        lux, luy = roi_d["lux"], roi_d["luy"]

        # ROI row
        ax_top = axes[0][col]
        ax_top.imshow(roi_img)
        ax_top.set_title(
            f"{specimen}\nlux={lux}  luy={luy}\n{roi_h}×{roi_w}px",
            fontsize=8, color="white", pad=4,
        )
        ax_top.axis("off")
        for spine in ax_top.spines.values():
            spine.set_edgecolor("#e07b54")
            spine.set_linewidth(2)
            spine.set_visible(True)

        # Padded row
        ax_bot = axes[1][col]
        ax_bot.imshow(pad_img)
        ax_bot.set_title(
            f"{pad_h}×{pad_w}px  (padded)",
            fontsize=8, color="white", pad=4,
        )
        ax_bot.axis("off")
        for spine in ax_bot.spines.values():
            spine.set_edgecolor("#5b9bd5")
            spine.set_linewidth(2)
            spine.set_visible(True)

    # Row labels on the left
    for row_idx, (label, color) in enumerate(zip(row_labels, label_colors)):
        fig.text(
            0.01, 0.75 - row_idx * 0.5, label,
            va="center", ha="left", fontsize=11,
            color=color, fontweight="bold", rotation=90,
        )

    plt.subplots_adjust(wspace=0.06, hspace=0.25)
    out = OUT_DIR / f"{cls.lower()}_pairs.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out.relative_to(ROOT)}")

print("Done.")
