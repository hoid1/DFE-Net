#!/usr/bin/env python3
"""
Rebuild the image folders of the fixed split from an original copy of NEU-DET
(only needed if the repository was cloned without images).

    python tools/prepare_neudet.py --src /path/to/NEU-DET

Images are matched by file name (e.g. crazing_1.jpg); YOLO labels are already in data/NEU-DET/labels.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = {".jpg", ".jpeg", ".png", ".bmp"}

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
a = ap.parse_args()
index = {p.name: p for p in Path(a.src).rglob("*") if p.suffix.lower() in EXT}
missing = []
for split in ("train", "val"):
    dst = ROOT / "data/NEU-DET/images" / split
    dst.mkdir(parents=True, exist_ok=True)
    for name in (ROOT / "data/NEU-DET/splits" / f"{split}.txt").read_text().split():
        if name in index:
            shutil.copy2(index[name], dst / name)
        else:
            missing.append(f"{split}/{name}")
print(f"done; {len(missing)} images not found" + (f": {missing[:10]} ..." if missing else ""))
