# DFE-Net: Dual-Stream Gated Cross-Modulation and Spatial-Frequency Detail Compensation for Steel Surface Defect Detection

Code, trained weights, the exact data split, training logs and a step-by-step reproduction procedure for the manuscript

> M. Shi, J. Zhao, W. Cai, Z. Xue. *DFE-Net: Dual-Stream Gated Cross-Modulation and Spatial-Frequency Detail Compensation for Steel Surface Defect Detection.* Submitted to *Electronics* (MDPI), manuscript electronics-4640316, 2026.

DFE-Net is built on YOLO11n (Ultralytics 8.4.11) and adds three components: DA-GCM (feature extraction), SFDP (neck) and SSP-Head (detection head). The correspondence between the names used in the paper and the class names used in the code is given in [MODULES.md](MODULES.md).

## Repository contents

```
configs/                 model YAMLs (exported from the released checkpoints) and training hyper-parameters
  yolo11n-dfe-net.yaml       DFE-Net
  yolo11n-baseline.yaml      YOLO11n baseline
  train_args_*.yaml          every training argument of the released runs
  original/                  original YAML files used during the experiments
weights/                 released checkpoints (seed 0; Tables 4, 7-10, 12-13 of the paper)
data/NEU-DET/            fixed 1440/360 split: images, YOLO labels, file lists, per-class statistics
runs/                    raw training logs (results.csv, args.yaml) of all experiments; runs_index.csv summarises them
tools/                   val.py, train.py, verify_release.py, prepare_neudet.py
ultralytics/             Ultralytics 8.4.11 with the DFE-Net modules (AGPL-3.0)
env/                     exact software environment of the experiments (pip freeze, GPU/driver info)
reproduce.sh             one-command reproduction
release_manifest.json    hashes, parameter counts and provenance of the released files
```

## 1. Installation

```bash
conda create -n dfenet python=3.11 -y && conda activate dfenet
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129   # match your CUDA
pip install -r requirements.txt
```

Do not `pip install ultralytics`: the scripts use the bundled `ultralytics/` folder, which contains the DFE-Net modules. Environment used for the paper:

```
python: 3.11.15
platform: Linux-5.15.0-135-generic-x86_64-with-glibc2.35
torch: 2.8.0+cu129
cuda (torch): 12.9
cudnn: 91002
gpu: NVIDIA GeForce RTX 3090
```

## 2. Data

NEU-DET (Song & Yan, 2013; He et al., 2020) is a public dataset of 1800 grayscale 200 × 200 images of hot-rolled steel strip surface defects in six classes, originally distributed by Northeastern University at <https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm>. This repository contains the fixed split used for **all** experiments of the paper, with labels converted to YOLO format (class x_center y_center width height, normalised). No images were added, removed or modified. The file lists are in `data/NEU-DET/splits/` and the per-class counts in `data/NEU-DET/split_statistics.csv` (Table 3 of the paper):

| Class | Train images | Val images | Train boxes | Val boxes |
|---|---|---|---|---|
| crazing | 240 | 60 | 555 | 133 |
| inclusion | 299 | 83 | 775 | 235 |
| patches | 275 | 67 | 694 | 186 |
| pitted_surface | 240 | 61 | 339 | 93 |
| rolled-in_scale | 240 | 60 | 501 | 127 |
| scratches | 240 | 60 | 440 | 108 |
| **Total** | 1440 | 360 | 3304 | 882 |

If you cloned a version without images, rebuild them from an original NEU-DET copy with `python tools/prepare_neudet.py --src /path/to/NEU-DET`. The dataset remains subject to the terms of its original authors; please cite them when using it.

## 3. Reproduce the reported results

```bash
python tools/verify_release.py      # weight hashes, split counts (Table 3), models rebuilt from YAML
python tools/val.py --model all     # re-evaluates both checkpoints and prints reproduced vs. paper values
```

`tools/val.py` uses the evaluation protocol of the paper: 640 × 640 input, FP32, confidence threshold 0.001, NMS IoU 0.7, batch size 32, validation split. Expected values (Tables 4 and 9):

| Model | P | R | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|---|
| YOLO11n | 68.2 | 68.8 | 72.6 | 39.1 | 41.1 | 2.58 | 6.3 |
| DFE-Net | 73.5 | 65.7 | 75.4 | 40.8 | 42.6 | 2.24 | 7.4 |

Accuracy should match up to rounding; FPS depends on the GPU (RTX 3090 in the paper).

Released weights:

| Model | File | Params | SHA-256 |
|---|---|---|---|
| dfe-net | `weights/dfe-net_best.pt` | 2,246,720 | `e2d8e5b083065be55b5e8972d786db2663d3bc2247d588aed3465fa37288f271` |
| yolo11n | `weights/yolo11n_best.pt` | 2,591,010 | `728caace86e9e67abdfe8115f73ddf827c14ea9bddcde502bad0ef588f9c4cdf` |

## 4. Train from scratch

```bash
python tools/train.py --model yolo11n --seed 0
python tools/train.py --model dfe-net --seed 0
# five-seed study (Table 11)
for s in 0 11 42 123 3407; do python tools/train.py --model dfe-net --seed $s; done
```

`tools/train.py` reads every hyper-parameter from `configs/train_args_*.yaml`, i.e. from the checkpoints themselves (SGD, lr0 0.01, 300 epochs, batch 16, mosaic for all epochs, no pretrained weights; see Table 1 of the paper). The checkpoint with the highest mAP@0.5 + mAP@0.5:0.95 on the validation split is kept as `best.pt`. Because of non-deterministic GPU kernels, a retrained model will differ slightly from the released one; Table 11 reports a seed-to-seed standard deviation of 0.64–0.81 pp in mAP@0.5. `bash reproduce.sh --train` runs the complete procedure.

## 5. Raw training data

`runs/` holds the per-epoch logs (`results.csv`: losses, precision, recall, mAP@0.5, mAP@0.5:0.95, learning rates) and the full arguments (`args.yaml`) of the experiments in the paper, including the ablations and the multi-seed runs. `runs/runs_index.csv` lists for every run the epoch with the highest fitness and its metrics. The curves of Figure 14 are plotted from the `results.csv` files of the two released models.

## 6. License and acknowledgements

The code is released under AGPL-3.0, inherited from [Ultralytics](https://github.com/ultralytics/ultralytics). DFE-Net builds on ideas and public implementations cited in the paper, including FCM/FBRT-YOLO, SPD-Conv, the omni-kernel network, the features-fused pyramid neck and the modulation fusion module. NEU-DET is the property of its original authors.

## 7. Citation

```bibtex
@article{shi2026dfenet,
  title   = {DFE-Net: Dual-Stream Gated Cross-Modulation and Spatial-Frequency Detail Compensation for Steel Surface Defect Detection},
  author  = {Shi, Mingshang and Zhao, Jing and Cai, Weibin and Xue, Zhipeng},
  journal = {Electronics},
  year    = {2026},
  note    = {Under review, manuscript electronics-4640316}
}
```

Contact: zhaojing.83@163.com
