"""Shared helpers: always use the bundled `ultralytics/` package and run from the repository root."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # bundled package takes precedence over any pip-installed ultralytics
os.chdir(ROOT)                  # data.yaml uses paths relative to the repository root

DATA = ROOT / "data/NEU-DET/data.yaml"
MODELS = {
    "dfe-net": {"cfg": ROOT / "configs/yolo11n-dfe-net.yaml", "weights": ROOT / "weights/dfe-net_best.pt",
                "args": ROOT / "configs/train_args_dfe-net.yaml"},
    "yolo11n": {"cfg": ROOT / "configs/yolo11n-baseline.yaml", "weights": ROOT / "weights/yolo11n_best.pt",
                "args": ROOT / "configs/train_args_yolo11n.yaml"},
}
# Single-run values reported in Tables 4 and 9 of the manuscript (seed 0, NEU-DET validation split).
PAPER = {
    "dfe-net": {"P": 73.5, "R": 65.7, "mAP50": 75.4, "mAP75": 40.8, "mAP50-95": 42.6, "params_M": 2.24, "GFLOPs": 7.4},
    "yolo11n": {"P": 68.2, "R": 68.8, "mAP50": 72.6, "mAP75": 39.1, "mAP50-95": 41.1, "params_M": 2.58, "GFLOPs": 6.3},
}
