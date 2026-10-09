# Paper names ↔ code names

| Paper (Section) | Code class | Where it is used |
|---|---|---|
| DA-GCM, C3k2 with DA-GCM (3.2) | `DGCM`, C3k2 variant with DGCM bottleneck | all eight C3k2 blocks |
| DBRConv, P2 → P3 injection (3.3.1) | `SPDConv` | P3 fusion node |
| DSFC with DESF / FGM (3.3.2) | `CSPOmniKernel`, `OmniKernel`, `FGM` | after the P3 fusion node |
| DRPU (3.3.1) | soft nearest-neighbour upsampling of the RFPN neck | top-down path |
| DCSD (3.3.1) | group-shuffle downsampling of the RFPN neck | bottom-up path |
| DASF (3.3.3) | `MFM` | every fusion node |
| SSP-Head (3.4) | `Detect_Efficient` | detection head |

Layer types actually present in the released DFE-Net checkpoint and the files that define them (generated automatically):

| Layer type | Source file |
|---|---|
| `Attention` | `ultralytics/nn/modules/block.py` |
| `C2PSA` | `ultralytics/nn/modules/block.py` |
| `C3k2_Block` | `ultralytics/nn/extra_modules/block/CSPBlock.py` |
| `C3k_Block` | `ultralytics/nn/extra_modules/block/CSPBlock.py` |
| `CSPOmniKernel` | `ultralytics/nn/extra_modules/neck/SOEP.py` |
| `Conv` | `ultralytics/nn/modules/conv.py` |
| `DFL` | `ultralytics/nn/modules/block.py` |
| `DGCM` | `ultralytics/nn/extra_modules/module/DGCM.py` |
| `Detect_Efficient` | `ultralytics/nn/extra_modules/head/Efficient.py` |
| `DualPoolChannelAttention` | `ultralytics/nn/extra_modules/module/DGCM.py` |
| `FGM` | `ultralytics/nn/extra_modules/neck/SOEP.py` |
| `GSConvE` | `ultralytics/nn/extra_modules/neck/RFPN.py` |
| `MFM` | `ultralytics/nn/extra_modules/featurefusion/mfm.py` |
| `OmniKernel` | `ultralytics/nn/extra_modules/neck/SOEP.py` |
| `PSABlock` | `ultralytics/nn/modules/block.py` |
| `SNI` | `ultralytics/nn/extra_modules/neck/RFPN.py` |
| `SPDConv` | `ultralytics/nn/extra_modules/downsample/SPDConv.py` |
| `SPPF` | `ultralytics/nn/modules/block.py` |
| `StripSpatialAttention` | `ultralytics/nn/extra_modules/module/DGCM.py` |
