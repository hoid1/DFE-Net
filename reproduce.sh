#!/usr/bin/env bash
# One-command reproduction of the main results of the manuscript.
#   bash reproduce.sh            # checks + evaluation of the released weights (a few minutes)
#   bash reproduce.sh --train    # additionally retrain both models from scratch (seed 0, 300 epochs each)
set -e
cd "$(dirname "$0")"
python tools/verify_release.py
python tools/val.py --model all
if [ "$1" == "--train" ]; then
  python tools/train.py --model yolo11n --seed 0
  python tools/train.py --model dfe-net --seed 0
  python tools/val.py --model dfe-net --weights runs_local/train/dfe-net_seed0/weights/best.pt
  python tools/val.py --model yolo11n --weights runs_local/train/yolo11n_seed0/weights/best.pt
fi
