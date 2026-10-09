#!/usr/bin/env python3
"""
Train DFE-Net or the YOLO11n baseline from scratch with exactly the hyper-parameters
stored in the released checkpoints (configs/train_args_*.yaml).

    python tools/train.py --model dfe-net --seed 0
    python tools/train.py --model yolo11n --seed 0
    for s in 0 11 42 123 3407; do python tools/train.py --model dfe-net --seed $s; done   # Table 11

Exact bit-level reproduction is not guaranteed across GPUs / CUDA / cuDNN versions;
expect run-to-run differences of the order reported in Table 11 (std 0.6-0.8 pp mAP@0.5).
"""
import argparse
import warnings

import yaml

from _common import DATA, MODELS, ROOT

warnings.filterwarnings("ignore")
from ultralytics import YOLO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=None, help="override (e.g. 1 for a smoke test)")
    a = ap.parse_args()

    cfg = MODELS[a.model]
    args = yaml.safe_load(open(cfg["args"])) or {}
    args.update(seed=a.seed, device=a.device, data=str(DATA), project=str(ROOT / "runs_local/train"),
                name=f"{a.model}_seed{a.seed}", exist_ok=False, pretrained=False)
    if a.epochs:
        args["epochs"] = a.epochs
    for k in ("model", "save_dir", "resume", "mode"):
        args.pop(k, None)
    print("training arguments:", {k: args[k] for k in sorted(args)})
    model = YOLO(str(cfg["cfg"]))
    model.train(**args)


if __name__ == "__main__":
    main()
