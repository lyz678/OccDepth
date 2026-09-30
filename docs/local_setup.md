# 本地 PyTorch 环境、SemanticKITTI 08 与运行说明

## 已适配环境

本次在 Conda `pytorch` 中验证：Python 3.10.18、torch 2.7.1+cu128、
torchvision 0.22.1+cu128、NumPy 2.1.2、timm 1.0.16、
pytorch-lightning 2.5.6、torchmetrics 1.8.2、Kornia 0.8.1、Numba 0.61.2。
不再需要 MMDetection、MMCV 或 MMDetection3D。

```bash
cd /home/lyz/OccDepth
conda activate pytorch
python -m pip install -r requirements.txt -c constraints-local.txt
# 测试工具（不是运行依赖）
python -m pip install pytest
```

`constraints-local.txt` 保护本机现有 Torch/torchvision/torchaudio/NumPy，
不是从零安装 CUDA PyTorch 的通用配置。可视化可选依赖单列于
`requirements-visualization.txt`，不影响训练和评估。

## 已下载的数据与模型

| 内容 | 本地位置 | 状态 |
|---|---|---|
| 指定作者权重 | `trained_models/kitti_multicam_flospdepth_crp_stereodepth_cascadecls_2080ti_mIoU12.8.ckpt` | 491,479,712 字节；严格加载成功 |
| 左右彩色图像 | `data/semantic_kitti/dataset/sequences/08/image_2`、`image_3` | 左右各 4,071 张 |
| 标定、时间戳 | 同序列目录的 `calib.txt`、`times.txt` | 完整标定含 `Tr` |
| 体素数据 | 同序列目录的 `voxels/` | 815 帧，每帧 `.bin/.label/.invalid/.occluded` |
| 预处理标签 | `data/kitti_semantic_preprocess/labels/08/` | 815 × 2 个尺度，共 1,630 文件 |
| 首帧预测 | `output/kitti/08/000000.pkl` | `uint16`，形状 `(256,256,32)` |

权重 SHA-256：

```text
8231cc38b9aaae94ced3c3d410871d137df572cac109e3b6284c523d781b6076
```

**作者提供的双目深度 ZIP 只有 00–07、09、10，没有 08。**
没有伪造或替代这部分数据。项目的 KITTI 验证/测试 dataloader 原本就不读取
深度监督数据，因此 08 评估和推理无需该文件；训练集深度不在本次下载范围。
来源检查结果见 `data/downloads/depth-08-manifest.json`。

下载来源为仓库 README 指向的作者 Google Drive、KITTI 官方 S3 和
SemanticKITTI 官方站点。`data/downloads/*-08-manifest.json` 保存各文件大小与
原 ZIP CRC32；`verification.json` 保存完整性检查与权重哈希。
数据、模型、预测和运行日志均被 Git 忽略。

## 直接运行

在项目根目录使用模块方式运行，避免 `PYTHONPATH` 问题：

```bash
cd /home/lyz/OccDepth
conda activate pytorch
export OCCDEPTH_ROOT="$PWD"
export DATA_CONFIG="$PWD/occdepth/config/semantic_kitti/kitti08_local.yaml"

# 一个验证批次；整数 1 表示一个 batch，1.0 表示全部。
python -m occdepth.scripts.eval limit_test_batches=1

# 完整 815 帧评估（本次未执行，不将单批结果视为论文精度复现）。
python -m occdepth.scripts.eval limit_test_batches=1.0

# 生成首个 batch 的预测；去掉 +max_batches=1 则生成全部验证集预测。
python -m occdepth.scripts.generate_output +max_batches=1

# CPU 可用，但完整尺寸模型会显著变慢。
python -m occdepth.scripts.eval n_gpus=0 limit_test_batches=1

# 标签已生成；需要时可重跑，已存在的标签不会重复计算。
python -m occdepth.data.semantic_kitti.preprocess

# 文件、标签及原 ZIP CRC32 完整性复查。
python tools/verify_kitti08.py --crc

# 不下载预训练参数，使用小网格的真实模型训练/评估测试。
python -m pytest -q tests
```

本地配置不覆盖发布配置；设置 `OCCDEPTH_ROOT` 可迁移项目位置。
`n_gpus=0` 为 CPU，`1` 为单 GPU，`>1` 为 DDP；请求不可用的 GPU 会明确报错。
预测脚本是单进程单卡/CPU 工具，使用 `n_gpus=1` 或 `0`；批大小以实际输入为准。

加载完整模型时，统一通过 `ckpt` 指定可信的本地 Lightning checkpoint，
关闭额外 backbone 权重下载，保持 `strict=True`。旧 checkpoint 包含 pickle
配置元数据，因此只在明确的完整 checkpoint 加载入口使用 `weights_only=False`。
作者的旧 Lightning 1.4.9 checkpoint 会触发自动格式迁移提示及旧 callback
键冲突提示；模型参数全部匹配，原始文件不被改写。这不等于验证了旧训练
callback 状态可以无损恢复；新版本 checkpoint 的保存与继续训练已测试。

### 训练

本次只下载了 08，不能用它冒充训练集。准备 00–07、09、10 的图像、体素、
预处理标签和作者双目深度后，使用原训练配置覆盖路径：

```bash
export DATA_CONFIG="$PWD/occdepth/config/semantic_kitti/multicam_flospdepth_crp_stereodepth_cascadecls_2080ti.yaml"
python -m occdepth.scripts.train \
  data_root="$PWD/data/semantic_kitti" \
  data_preprocess_root="$PWD/data/kitti_semantic_preprocess" \
  data_stereo_depth_root="$PWD/data/KITTI_Odometry_Stereo_Depth" \
  n_gpus=1
```

默认训练使用预训练 EfficientNet，首次可能下载 timm 权重。
`+pretrained_backbone=false` 可禁止该下载（从随机 backbone 开始训练）。
已有训练 checkpoint 可通过 `ckpt=/absolute/path/last.ckpt` 恢复；
未指定时仍自动寻找当前实验目录下的 `checkpoints/last.ckpt`。
`+limit_train_batches=1 +limit_val_batches=1 max_epochs=1` 可限制训练冒烟运行。
提交文件生成入口保留，但其 11–21 测试数据不在本次下载范围。

## 下载复用与空间预算

指定权重也可从作者链接重新下载（现有完整文件无需重复下载）：

```bash
python -m pip install 'gdown>=6,<7'
python -m gdown 'https://drive.google.com/uc?id=1MGJ_HZcuW5UpULpOeJV0M5ZrT-98j7OE' \
  -O trained_models/kitti_multicam_flospdepth_crp_stereodepth_cascadecls_2080ti_mIoU12.8.ckpt --continue
```

仅下载 08 数据：

```bash
python tools/download_kitti08.py color
python tools/download_kitti08.py calib
python tools/download_kitti08.py voxels
```

下载器直接读取官方 ZIP 的中央目录并通过 HTTP Range 获取 08 成员，
不落地约 69 GB 的完整图像包。已完成的文件按大小和 CRC 检查后跳过，
未完成成员使用 `.part` 重试；至少保留 20 GiB 磁盘空间。
图像源仅提取 `image_2/image_3`，避免其中的简化标定覆盖独立标定包。
体素标签按名称对应 invalid mask，缺失文件不静默跳过。

## 本次验收（2026-09-30）

- 14 项测试通过：原生残差前向/反向对照，Conv/SE/MobileNet，冻结与梯度检查点，
  B3/B4/B5/B7 特征尺寸，投影不可见体素的有限梯度，下载 CRC/标定保护，
  KITTI 深度投影和 NYU 虚拟双目实际模型的训练、验证、测试、保存与继续训练。
- 真实作者 checkpoint：1,264 个 state-dict 项严格匹配，40,859,886 参数。
- 实际 GPU 单批评估成功，峰值已分配显存约 **1.52 GiB**；首帧预测文件已生成。
- 所有已下载图像、标定、体素文件 CRC 校验通过；预处理标签形状和类别范围通过。
- `pip check`、Python 编译检查和 `git diff --check` 通过。
- 未运行完整 08 精度复现、完整训练、多 GPU 实机测试；没有下载 NYU 或 KITTI 训练/测试序列。

测试日志：`data/downloads/eval-08-smoke.log`、`predict-08-smoke.log`、
`verification.log`。最初沙箱内看不到 GPU，但沙箱外已确认驱动和 CUDA 可用，
不需要重装驱动。
