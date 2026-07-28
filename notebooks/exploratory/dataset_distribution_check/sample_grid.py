"""
Overlap analysis + visual sample grid for old vs new dataset split.

Output (all written to the same directory as this script):
  - Prints a per-class overlap report to stdout
  - Saves three PNGs: benign_grid.png, malignant_grid.png, suspicious_grid.png
    Each PNG shows two 4x4 grids side-by-side:
      Left  — 16 random samples from AnnotatedDataSet_old
      Right — 16 random samples from AnnotatedDataSet new-only files
              (guaranteed not to appear in old)
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

SEED = 42
N_SAMPLES = 16
CLASSES = ["Benign", "Malignant", "Suspicious"]

ROOT = Path(__file__).resolve().parents[3]  # project root
DATA_RAW = ROOT / "data" / "raw"
OLD_DIR = DATA_RAW / "AnnotatedDataSet_old"
NEW_DIR = DATA_RAW / "AnnotatedDataSet"
OUT_DIR = Path(__file__).parent

random.seed(SEED)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def png_names(directory: Path) -> set[str]:
    return {f.name for f in directory.iterdir() if f.suffix == ".png"}


# ---------------------------------------------------------------------------
# overlap report
# ---------------------------------------------------------------------------

print("=" * 64)
print("OVERLAP ANALYSIS REPORT")
print("=" * 64)

overlap: dict[str, dict] = {}
for cls in CLASSES:
    old = png_names(OLD_DIR / cls)
    new = png_names(NEW_DIR / cls)
    shared   = old & new
    old_only = old - new   # in old but MISSING from new  ← the "weird" ones
    new_only = new - old   # genuine additions in new

    overlap[cls] = dict(old=old, new=new, shared=shared,
                        old_only=old_only, new_only=new_only)

    pct_shared = 100 * len(shared) / len(old) if old else 0
    print(f"\n{cls}:")
    print(f"  Old total          : {len(old):>5}")
    print(f"  New total          : {len(new):>5}")
    print(f"  Shared             : {len(shared):>5}  ({pct_shared:.1f}% of old retained in new)")
    print(f"  Old-only (missing) : {len(old_only):>5}  ← files in old but gone from new")
    print(f"  New-only (added)   : {len(new_only):>5}  ← genuine new additions")

print("\n" + "=" * 64)


# ---------------------------------------------------------------------------
# sample grids
# ---------------------------------------------------------------------------

def load(path: Path) -> "Image.Image":
    return Image.open(path).convert("RGB")


for cls in CLASSES:
    d = overlap[cls]

    if len(d["old"]) < N_SAMPLES:
        raise ValueError(f"{cls}: not enough old samples ({len(d['old'])} < {N_SAMPLES})")
    if len(d["new_only"]) < N_SAMPLES:
        raise ValueError(f"{cls}: not enough new-only samples ({len(d['new_only'])} < {N_SAMPLES})")

    old_sample     = random.sample(sorted(d["old"]),     N_SAMPLES)
    new_only_sample = random.sample(sorted(d["new_only"]), N_SAMPLES)

    # 4 rows x 8 cols — left block = old, right block = new-only
    fig, axes = plt.subplots(4, 8, figsize=(18, 9))
    fig.patch.set_facecolor("#1a1a1a")

    title = (
        f"{cls}   |   "
        f"Old: {len(d['old'])}   "
        f"New-only: {len(d['new_only'])}   "
        f"Shared: {len(d['shared'])}   "
        f"Old-only (missing from new): {len(d['old_only'])}"
    )
    fig.suptitle(title, fontsize=11, color="white", y=1.01, fontweight="bold")

    # group labels
    fig.text(0.255, 1.035, "Old", ha="center", fontsize=13,
             color="#e07b54", fontweight="bold")
    fig.text(0.745, 1.035, "New-Only", ha="center", fontsize=13,
             color="#5b9bd5", fontweight="bold")

    # render thumbnails
    for i, fname in enumerate(old_sample):
        row, col = divmod(i, 4)
        ax = axes[row][col]
        ax.imshow(load(OLD_DIR / cls / fname))
        ax.axis("off")

    for i, fname in enumerate(new_only_sample):
        row, col = divmod(i, 4)
        ax = axes[row][col + 4]
        ax.imshow(load(NEW_DIR / cls / fname))
        ax.axis("off")

    # vertical separator
    sep = plt.Line2D(
        [0.503, 0.503], [0.01, 0.99],
        transform=fig.transFigure,
        color="#666666", linewidth=1.5, linestyle="--",
    )
    fig.add_artist(sep)

    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    out_path = OUT_DIR / f"{cls.lower()}_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path.relative_to(ROOT)}")

print("\nDone.")
