# OccDepth: A Depth-aware Method for 3D Semantic Occupancy Network 

![](https://img.shields.io/badge/Ranked%20%231-Camera--Only%203D%20Semantic%20Scene%20Completion%20on%20SemanticKITTI-blue "")

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/occdepth-a-depth-aware-method-for-3d-semantic/3d-semantic-scene-completion-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-scene-completion-on-semantickitti?p=occdepth-a-depth-aware-method-for-3d-semantic)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/occdepth-a-depth-aware-method-for-3d-semantic/3d-semantic-scene-completion-on-nyuv2)](https://paperswithcode.com/sota/3d-semantic-scene-completion-on-nyuv2?p=occdepth-a-depth-aware-method-for-3d-semantic)
# News
- **2023/03/30** Release trained models on GeForce RTX 2080 Ti.
- **2023/02/28** Initial code release. Both Stereo images and RGB-D images inputs are supported.
- **2023/02/28** Paper released on [Arxiv](https://arxiv.org/abs/2302.13540).
- **2023/02/17** Demo release.

# Abstract
In this paper, we propose the first stereo SSC method named OccDepth, which fully exploits implicit depth information from stereo images (or RGBD images) to help the recovery of 3D geometric structures. The Stereo Soft Feature Assignment (Stereo-SFA) module is proposed to better fuse 3D depth-aware features by implicitly learning the correlation between stereo images. In particular, when the input are RGBD image, a virtual stereo images can be generated through original RGB image and depth map. Besides, the Occupancy Aware Depth (OAD) module is used to obtain geometry-aware 3D features by knowledge distillation using pre-trained depth models.

# Video Demo

Mesh results compared with ground truth on KITTI-08:
<p align="center">
<img src="./assets/demo.gif" alt="video loading..." />
</p>
Voxel results compared with ground truth on KITTI-08:
<p align="center">
<img src="./assets/demo_voxel.gif" alt="video loading..." />
</p>
Full demo videos can be downloaded via `git lfs pull`, the demo videos are saved as "assets/demo.mp4" and "assets/demo_voxel.mp4". 

# Results
## Trained models

The trained models on GeForce RTX 2080 Ti are provided:
| Config| dataset |IoU| mIoU |  Download |
| :---: | :---: | :---: | :---: | :---:|
| [config](occdepth/config/semantic_kitti/multicam_flospdepth_crp_stereodepth_cascadecls_2080ti.yaml) | SemanticKITTI | 41.60| 12.84|[model](https://drive.google.com/file/d/1MGJ_HZcuW5UpULpOeJV0M5ZrT-98j7OE/view?usp=share_link) |
| [config](occdepth/config/NYU/multicam_flosp_crp_stereodepth_cascadecls_2080ti.yaml) | NYUv2 | 49.23| 29.34|[model](https://drive.google.com/file/d/1tBKB-J6NAxDTRTOE1hwAacRmwu57q8L8/view?usp=share_link)|

Note: If you want to get better results, you should set `share_2d_backbone_gradient = false`, `backbone_2d_name = tf_efficientnet_b7_ns` and `feature = feature_2d_oc = 64 (SemanticKITTI)` which needs more GPU memory.
## Qualitative Results
<div align="center">
<img width=374 src="./assets/result1-1.png"/><img width=400 src="./assets/result1-2.png"/>


Fig. 1: RGB based Semantic Scene Completion with/without depth-aware. (a) Our proposed OccDepth method can detect smaller and farther objects. (b) Our proposed OccDepth method complete road better.
</div>

## Quantitative results on SemanticKITTI

<div align="center">
Table 1. Performance on SemanticKITTI (hidden test set). 

|Method            |Input        | SC  IoU       | SSC mIoU       |
|:----------------:|:----------:|:--------------:|:--------------:|
| **2.5D/3D**      |            |                |                |
| LMSCNet(st)   | OCC        | 33.00          | 5.80           |
| AICNet(st)    | RGB, DEPTH | 32.8           | 6.80           |
| JS3CNet(st)   | PTS        | 39.30          | 9.10           |
| **2D**           |            |                |                |
| MonoScene        | RGB        | 34.16          | 11.08          |
| MonoScene(st) | Stereo RGB | 40.84          | 13.57          |
| OccDepth (ours)  | Stereo RGB | **45.10**      | **15.90**      |
</div>
The scene completion (SC IoU) and semantic scene completion (SSC mIoU) are reported for modified baselines (marked with "st") and our OccDepth.

## Detailed results on SemanticKITTI.
<div align="center">
<img src="./assets/result2.png"/>
</div>

## Compared with baselines.
<div align="center">
<img width=400 src="./assets/result3.png"/>
</div>
Baselines of 2.5D/3D-input methods. ”∗
” means results are cited from MonoScene. ”/”
means missing results

# Usage

## Environment

This checkout runs without MMDetection/MMCV. The validated local environment is
Python 3.10, PyTorch 2.7.1+CUDA 12.8, torchvision 0.22.1, NumPy 2.1.2 and
PyTorch Lightning 2.5.6.

```bash
conda activate pytorch
python -m pip install -r requirements.txt -c constraints-local.txt
```

Keep the existing CUDA-enabled PyTorch installation. `constraints-local.txt`
protects the versions installed on this machine; it is not a generic fresh
PyTorch installation recipe. Visualization packages are optional and listed
separately in `requirements-visualization.txt`.

**Dataset files and checkpoints are not included in this repository.** Follow
the download steps below before running evaluation. See also
[local setup, verified results and runnable commands](docs/local_setup.md).
The author depth archive does not contain sequence 08; validation does not
require depth supervision. The quick start prepares only validation sequence 08,
not the training/test splits.

```bash
export OCCDEPTH_ROOT="$PWD"
export DATA_CONFIG="$PWD/occdepth/config/semantic_kitti/kitti08_local.yaml"
python -m occdepth.scripts.eval limit_test_batches=1
```

## Preparing

### SemanticKITTI

#### Quick start: download validation sequence 08

Run these commands from the repository root after installing the environment
dependencies. The downloader reads the official ZIP directory and fetches only
sequence 08 using HTTP Range requests; it does **not** download the entire
approximately 69 GB color-image archive.

```bash
git clone https://github.com/lyz678/OccDepth.git
cd OccDepth
conda activate pytorch
python -m pip install -r requirements.txt -c constraints-local.txt
python -m pip install 'gdown>=6,<7'

# Left/right images, full calibration and semantic-completion voxel labels.
python tools/download_kitti08.py color
python tools/download_kitti08.py calib
python tools/download_kitti08.py voxels

# Download the author's checkpoint, preserving its original filename.
mkdir -p trained_models
python -m gdown 'https://drive.google.com/uc?id=1MGJ_HZcuW5UpULpOeJV0M5ZrT-98j7OE' \
  -O trained_models/kitti_multicam_flospdepth_crp_stereodepth_cascadecls_2080ti_mIoU12.8.ckpt \
  --continue

# Generate full-resolution and 1/8-resolution semantic labels for 08 only.
export OCCDEPTH_ROOT="$PWD"
export DATA_CONFIG="$PWD/occdepth/config/semantic_kitti/kitti08_local.yaml"
python -m occdepth.data.semantic_kitti.preprocess

# Check frame pairing, calibration, label shapes/classes, ZIP CRCs and model hash.
python tools/verify_kitti08.py --crc

# One validation batch; use limit_test_batches=1.0 for all 815 labeled frames.
python -m occdepth.scripts.eval limit_test_batches=1
```

Expected files:

```text
OccDepth/
├── data/
│   ├── semantic_kitti/dataset/sequences/08/
│   │   ├── image_2/           # 4,071 left images
│   │   ├── image_3/           # 4,071 right images
│   │   ├── calib.txt          # Must include P2, P3 and Tr
│   │   ├── times.txt
│   │   └── voxels/            # 815 frames: .bin/.label/.invalid/.occluded
│   ├── kitti_semantic_preprocess/labels/08/
│   │   ├── 000000_1_1.npy
│   │   └── 000000_1_8.npy     # Two scales for every labeled frame
│   └── downloads/             # Source URLs, per-file CRCs and verification reports
└── trained_models/
    └── kitti_multicam_flospdepth_crp_stereodepth_cascadecls_2080ti_mIoU12.8.ckpt
```

The prepared 08 data, labels and checkpoint occupy approximately **17 GB**.
The downloader additionally reserves **20 GiB** of free disk space. Rerun the
same download command after an interruption: completed files are checked by
size and CRC before being skipped; incomplete members are retried via `.part`
files. `--root /path/to/data` selects another data directory; in that case,
override `data_root` and `data_preprocess_root` for preprocessing/evaluation.
The verification tool's `--root` instead selects the project root containing
both `data/` and `trained_models/`.

**No 08 depth download is required.** The author's stereo-depth archive contains
only training sequences 00–07 and 09–10. The KITTI validation/test loader disables
depth supervision, so this checkpoint can evaluate sequence 08 without it.
Do not substitute training-sequence depth files for missing validation depth.

#### Full training/test data and official sources

The quick-start downloader is intentionally limited to sequence 08. For full
training or test-set prediction, download the corresponding archives manually:

| Resource | Official/author source | Purpose |
|---|---|---|
| Color stereo images | [KITTI color archive](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip) | Left/right images; full archive is about 69 GB |
| Full calibration | [KITTI calibration archive](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip) | Camera projection and LiDAR-to-camera transform |
| Voxel data | [SemanticKITTI voxel archive](https://www.semantic-kitti.org/assets/data_odometry_voxels.zip) | Scene-completion input, labels and masks |
| Training stereo depth | [Author's depth archive](https://drive.google.com/file/d/1eJPJ1niczagkJfEv21_RdvYBDUbpaQ0w/view) | Depth supervision for 00–07 and 09–10 |

Use 00–07 and 09–10 for training, 08 for validation, and 11–21 for test
prediction (test ground-truth semantics are not public). Merge the image and
voxel archives under `data/semantic_kitti/dataset/sequences/`. Extract the
**full calibration archive last**, since image archives may contain a reduced
`calib.txt` without `Tr`. Extract the training depth archive so that files follow
`data/KITTI_Odometry_Stereo_Depth/dataset/sequences/<sequence>/depth/<frame>.png`.
Budget disk space separately for full archives, extracted data and generated
labels; the 17 GB estimate above applies only to sequence 08.

To preprocess all training/validation sequences, select the original training
configuration rather than the 08-only preset:

```bash
export DATA_CONFIG="$PWD/occdepth/config/semantic_kitti/multicam_flospdepth_crp_stereodepth_cascadecls_2080ti.yaml"
python -m occdepth.data.semantic_kitti.preprocess \
  data_root="$PWD/data/semantic_kitti" \
  data_preprocess_root="$PWD/data/kitti_semantic_preprocess"
```

Configure `data_stereo_depth_root` before training. See
[training and checkpoint-resume commands](docs/local_setup.md#训练).


### NYUv2
- Download NYUv2 dataset
    - [NYUv2](https://www.rocq.inria.fr/rits_files/computer-vision/monoscene/nyu.zip)

- Preprocessed NYUv2 data
    ``` bash
    cd OccDepth/
    python occdepth/data/NYU/preprocess.py data_root="/path/to/NYU/depthbin"
    data_preprocess_root="/path/to/NYU/preprocess/folder"
    ```
### Settings
1. Setting `DATA_LOG`, `DATA_CONFIG` in `env_{dataset}.sh`, examples:
    ``` bash
    ##examples
    export DATA_LOG=$workdir/logdir/semanticKITTI
    export DATA_CONFIG=$workdir/occdepth/config/semantic_kitti/multicam_flospdepth_crp_stereodepth_cascadecls_2080ti.yaml
    ```
2. Setting `data_root`, `data_preprocess_root` and `data_stereo_depth_root` in config file (occdepth/config/xxxx.yaml), examples:
    ``` yaml
    ##examples
    data_root: '/data/dataset/KITTI_Odometry_Semantic'
    data_preprocess_root: '/data/dataset/kitti_semantic_preprocess'
    data_stereo_depth_root: '/data/dataset/KITTI_Odometry_Stereo_Depth'
    ```

## Inference

``` bash
cd OccDepth/
source env_{dataset}.sh
## move the trained model to OccDepth/trained_models/occdepth.ckpt
## Single GPU prediction export, batch size 1
python -m occdepth.scripts.generate_output n_gpus=1 batch_size_per_gpu=1
```

## Evaluation
``` bash
cd OccDepth/
source env_{dataset}.sh
## move the trained model to OccDepth/trained_models/occdepth.ckpt
## 1 gpu and batch size on each gpu is 1
python -m occdepth.scripts.eval n_gpus=1 batch_size_per_gpu=1
```
## Training
``` bash
cd OccDepth/
source env_{dataset}.sh
## 4 gpus and batch size on each gpu is 1
python occdepth/scripts/train.py logdir=${DATA_LOG} n_gpus=4 batch_size_per_gpu=1
```
# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Acknowledgements
Our code is based on these excellent open source projects: 
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [SSC](https://github.com/waterljwant/SSC)
- [UVTR](https://github.com/dvlab-research/UVTR)
- [CaDDN](https://github.com/TRAILab/CaDDN)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)

Many thanks to them!

# Related Repos
* https://github.com/wzzheng/TPVFormer
* https://github.com/FANG-MING/occupancy-for-nuscenes
* https://github.com/nvlabs/voxformer

# Citation
If you find this project useful in your research, please consider cite:
```
@article{miao2023occdepth,
Author = {Ruihang Miao and Weizhou Liu and Mingrui Chen and Zheng Gong and Weixin Xu and Chen Hu and Shuchang Zhou},
Title = {OccDepth: A Depth-Aware Method for 3D Semantic Scene Completion},
journal = {arXiv:2302.13540},
Year = {2023},
}
```
# Contact
If you have any questions, feel free to open an issue or contact us at miaoruihang@megvii.com, huchen@megvii.com.
