"""
Pre-processes ROI images into information-free square 224x224 tiles.

For each image:
  1. Zero-pad the shorter side on the bottom/right to make it square.
     Original content always sits at the top-left of the square.
  2. Resize to 224x224 (LANCZOS) for ALL images.
     Since the image is already square, both axes scale equally — no aspect
     ratio distortion. Small images are upscaled, large ones downscaled.
     Every pixel in the output is real image content (no large black borders).

Source  : notebooks/data/ROI_by_class/{Benign,Malignant,Suspicious}/  (symlinks)
Output  : data/raw/ROI_square/{Benign,Malignant,Suspicious}/           (real PNGs)

Run from any directory — all paths resolve relative to this file.
"""

from pathlib import Path
from PIL import Image

TARGET = 224
CLASSES = ["Benign", "Malignant", "Suspicious"]

ROOT    = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "notebooks" / "data" / "ROI_by_class"
OUT_DIR = ROOT / "data" / "raw" / "ROI_square"


def pad_to_square_224(img: Image.Image) -> Image.Image:
    w, h = img.size

    # Step 1: zero-pad shorter side on bottom/right to make square
    sq = max(w, h)
    square = Image.new("RGB", (sq, sq), (0, 0, 0))
    square.paste(img, (0, 0))

    # Step 2: resize to TARGET x TARGET for all images
    if sq != TARGET:
        square = square.resize((TARGET, TARGET), Image.LANCZOS)

    return square   # exactly TARGET x TARGET


for cls in CLASSES:
    src_cls = SRC_DIR / cls
    out_cls = OUT_DIR / cls
    out_cls.mkdir(parents=True, exist_ok=True)

    files = sorted(src_cls.glob("*.png"))
    print(f"{cls}: processing {len(files)} images ...", end=" ", flush=True)

    for f in files:
        img = Image.open(f).convert("RGB")
        out_img = pad_to_square_224(img)
        out_img.save(out_cls / f.name)

    print(f"done → {out_cls.relative_to(ROOT)}")

print(f"\nAll done. Output: {OUT_DIR.relative_to(ROOT)}")
