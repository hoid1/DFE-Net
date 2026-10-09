#!/usr/bin/env python3
"""Integrity checks: weight hashes, dataset split (Table 3), and model construction from the YAML configs."""
import json
import sys
import hashlib
from pathlib import Path

from _common import DATA, MODELS, ROOT

PAPER_SPLIT = {"train_images": 1440, "val_images": 360, "train_boxes": 3304, "val_boxes": 882}
ok = True


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


man = json.loads((ROOT / "release_manifest.json").read_text())
for n, m in man["models"].items():
    p = ROOT / m["weights"]
    good = p.exists() and sha256(p) == m["sha256"]
    ok &= good
    print(f"[weights] {n:8s} {'OK' if good else 'MISSING / HASH MISMATCH'}  {p.name}")

root = ROOT / "data/NEU-DET"
for split in ("train", "val"):
    names = [x for x in (root / "splits" / f"{split}.txt").read_text().split() if x]
    imgs = [root / "images" / split / x for x in names]
    have_imgs = sum(p.exists() for p in imgs)
    boxes = sum(len([l for l in (root / "labels" / split / (Path(x).stem + ".txt")).read_text().splitlines() if l.strip()])
                for x in names)
    good = len(names) == PAPER_SPLIT[f"{split}_images"] and boxes == PAPER_SPLIT[f"{split}_boxes"]
    ok &= good and have_imgs == len(names)
    print(f"[data]    {split:5s} images={len(names)} (present on disk: {have_imgs}) boxes={boxes}  "
          f"{'OK' if good else 'MISMATCH with Table 3'}")
    if have_imgs < len(names):
        print("          -> images missing: run  python tools/prepare_neudet.py --src <original NEU-DET folder>")

from ultralytics.nn.tasks import DetectionModel  # noqa: E402
import yaml  # noqa: E402

nc = yaml.safe_load(open(DATA))["nc"]
for n, c in MODELS.items():
    m = DetectionModel(str(c["cfg"]), nc=nc, verbose=False)
    p = sum(x.numel() for x in m.parameters())
    good = p == man["models"][n]["params"]
    ok &= good
    print(f"[model]   {n:8s} built from {c['cfg'].name}: {p:,} parameters  {'OK' if good else 'MISMATCH'}")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
