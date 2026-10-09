#!/usr/bin/env python3
"""
Re-evaluate the released checkpoints with the protocol of the manuscript (Section 4.1/4.2.2):
640x640, FP32, conf 0.001, NMS IoU 0.7, batch 32, non-rectangular batches, validation split.

    python tools/val.py --model all
    python tools/val.py --model dfe-net --weights path/to/your_best.pt
"""
import argparse
import json
import warnings

from _common import DATA, MODELS, PAPER, ROOT

warnings.filterwarnings("ignore")
from ultralytics import YOLO  # noqa: E402


def evaluate(name, weights, device):
    model = YOLO(str(weights))
    m = model.val(data=str(DATA), split="val", imgsz=640, batch=32, conf=0.001, iou=0.7, rect=False,
                  half=False, device=device, plots=True, project=str(ROOT / "runs_local/val"), name=name,
                  exist_ok=True, verbose=True)
    res = {"P": m.box.mp * 100, "R": m.box.mr * 100, "mAP50": m.box.map50 * 100, "mAP75": m.box.map75 * 100,
           "mAP50-95": m.box.map * 100, "inference_ms_per_img": m.speed["inference"]}
    res["FPS"] = 1000.0 / res["inference_ms_per_img"] if res["inference_ms_per_img"] else None
    res["per_class_AP50"] = {model.names[c]: float(m.box.ap50[i]) * 100 for i, c in enumerate(m.box.ap_class_index)}
    res["per_class_AP50-95"] = {model.names[c]: float(m.box.ap[i]) * 100 for i, c in enumerate(m.box.ap_class_index)}
    try:
        model.fuse()
        info = model.info(verbose=False)  # (layers, params, gradients, GFLOPs) of the fused model
        res["params_M"], res["GFLOPs"] = info[1] / 1e6, info[3]
    except Exception as e:  # pragma: no cover
        print("model.info failed:", e)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", *MODELS])
    ap.add_argument("--weights", default=None, help="evaluate another checkpoint instead of the released one")
    ap.add_argument("--device", default="0", help="'0' for GPU 0, 'cpu' for CPU")
    a = ap.parse_args()

    names = list(MODELS) if a.model == "all" else [a.model]
    out = {}
    for n in names:
        w = a.weights or MODELS[n]["weights"]
        out[n] = evaluate(n, w, a.device)

    print("\n" + "=" * 92)
    print(f"{'model':10s} {'metric':10s} {'reproduced':>12s} {'paper':>10s} {'diff':>8s}")
    for n, r in out.items():
        for k in ("P", "R", "mAP50", "mAP75", "mAP50-95", "params_M", "GFLOPs"):
            if k in r and k in PAPER[n]:
                print(f"{n:10s} {k:10s} {r[k]:12.2f} {PAPER[n][k]:10.2f} {r[k] - PAPER[n][k]:+8.2f}")
        if r.get("FPS"):
            print(f"{n:10s} {'FPS':10s} {r['FPS']:12.1f} {'(hardware dependent; RTX 3090 in the paper)':>10s}")
    print("=" * 92)
    dst = ROOT / "runs_local/val/summary.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print("saved", dst)


if __name__ == "__main__":
    main()
