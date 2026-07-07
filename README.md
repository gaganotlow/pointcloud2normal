# point2normal — 内盖法向量估计训练

从点云估计燃油盖内盖的外法向量。包含两个网络：

| 网络 | 子目录 | 架构 | 参数量 |
|---|---|---|---|
| **NormalNet** | `normalnet/` | PointNet++ (3×SA + MLP) | ~1.5M |
| **MSECNet** | `msecnet/` | U-Net + 多尺度边缘条件 | ~10M |

## 目录结构

```
train_point2normal/
├── shared/                  # 两个网络共用
│   ├── cap_patch.py         # 旋钮局部面片提取
│   ├── make_normal_labels.py # RANSAC 标签生成
│   ├── knob_centers_all.py  # OBB 旋钮中心批处理
│   └── knob_centers.json    # 预计算的旋钮中心
├── normalnet/               # PointNet++ NormalNet
│   ├── train.py             # 训练
│   ├── precompute.py        # 批量预计算
│   ├── infer_normal.py      # 端到端推理编排
│   ├── _segobb.py           # YOLO 分割 + OBB
│   ├── s2_moge.py           # MoGe 深度估计
│   ├── _norminfer.py        # 法向量推理 + 可视化
│   └── pointnet2_utils.py   # PointNet++ SA 算子
├── msecnet/                 # MSECNet
│   ├── train_v1.py          # 训练 v1 (固定 1024 点，推荐)
│   ├── train_v2.py          # 训练 v2 (可变点数)
│   ├── precompute.py        # 批量预计算
│   └── MSECNet/             # MSECNet 模型库
│       ├── model/           # architectures, blocks
│       └── scripts/         # config, util, pointops
├── data/ -> (symlink)       # 训练数据
│   └── pcd_dataset_roi/     # 点云 ROI 数据集
├── models/ -> (symlink)     # YOLO 模型权重 + HF cache
├── output/                  # 推理输出
├── environment.yml          # conda 环境定义
└── setup.sh                 # 一键安装脚本
```

## 快速开始

### 1. 环境安装

```bash
cd /data2/shendu/code/ruoyu/train_point2normal
bash setup.sh
conda activate point2normal
```

setup.sh 会：
- 创建 conda 环境（PyTorch 2.6.0 + CUDA 12.4）
- 安装 MoGe、ultralytics、trimesh 等依赖
- 编译 MSECNet 的 pointops_cuda 算子
- 创建数据和模型的软链接

### 2. 生成 RANSAC 标签 —— 即训练数据

```bash
python shared/make_normal_labels.py data/pcd_dataset_roi \
    --radius 0.3 \
    --out shared/normal_labels_patch03.npz
```

### 3. NormalNet 训练

```bash
# Soft 课程训练（推荐）
python normalnet/train.py shared/normal_labels_patch03.npz data/pcd_dataset_roi \
    --steps 80000 --bs 32 --npoints 1024 --soft \
    --radius 0.3 --aug-deg 180 \
    --out normalnet/ckpt_normal
```

### 4. MSECNet 训练

```bash
# v1 (推荐，固定 1024 点)
python msecnet/train_v1.py shared/normal_labels_patch03.npz data/pcd_dataset_roi \
    --steps 20000 --bs 32 --npoints 1024 --soft \
    --radius 0.3 --aug-deg 45 \
    --out msecnet/ckpt_msecnet
```

### 5. 端到端推理

```bash
python normalnet/infer_normal.py path/to/image.png
```

产物在 `output/normal_pred_demo/`。

### 6. 批量预计算（生成网页标注的参考箭头）

三个预计算脚本，使用相同的数据源（`data/pcd_dataset_roi/` 全部 11135 个点云）：

```bash
# ① MoGe 法向量（绿色箭头）— 无需 GPU，纯 numpy，最快（~2 分钟）
python shared/precompute_moge_normal.py

# ② NormalNet 推理（用于分档排序 + v3 预标注）— GPU，~5 分钟
python normalnet/precompute.py normalnet/ckpt_normal/best.pt

# ③ MSECNet 推理（橙色箭头）— GPU，~10 分钟
python msecnet/precompute.py msecnet/ckpt_msecnet/best.pt
```

产物默认输出到 `shared/` 目录，网页标注工具会自动加载。也可通过第二个参数指定输出路径。

### 7. 网页标注工具

标注工具加载 `shared/` 下的预测结果在浏览器中显示为参考箭头：

| 文件 | 箭头颜色 | 来源 |
|---|---|---|
| `shared/v3_predictions.json` | 用于分档排序 | 步骤 6②（NormalNet） |
| `shared/msecnet_predictions.json` | 🟠 橙色 | 步骤 6③（MSECNet） |
| `shared/moge_norm_predictions.json` | 🟢 绿色 | 步骤 6①（MoGe，零训练） |
| `shared/normal_labels_full.npz` | 🔴 红色（可编辑） | 步骤 2（RANSAC 几何标签） |

```bash
# 终端 1: 启动标注服务器
python web_label/server.py --port 8765

# 终端 2: (可选但推荐) 启动 MoGe 后台 worker，生成全图稠密点云
# 首次运行会加载模型到显存 (~1.5GB)，之后每张图 2-4s
conda activate point2normal
HF_HOME=models/hf_cache CUDA_VISIBLE_DEVICES=1 python web_label/moge_worker.py

# 如果没启动 worker 但感觉加载慢，可以跳过全图云等待:
FULL_CLOUD=0 python web_label/server.py --port 8765
```

打开 `http://<host>:8765`，在浏览器中用 3D 视图人工校订法向量：
- 红色箭头 = 可编辑的当前法向量（拖红环旋转）
- 橙色箭头 = 模型预测 (MSECNet)，固定参考
- 绿色箭头 = MoGe 深度图法向量（零训练），固定参考
- 蓝色矩形 = 平面贴合检查器，贴到盖面上即正确
- 空格 = 保存并下一张，a/d = 上下翻页，x = 标异常
- r = 绕法向量旋转验证（正确时盖子不晃动）

标注结果保存在 `output/manual_normals.json`。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 0 | GPU 设备 |
| `HF_HOME` | models/hf_cache | HuggingFace 缓存目录 |
| `PCD_DATASET` | data/pcd_dataset_roi | 点云数据集路径 |
| `NORMALNET_CKPT` | normalnet/ckpt_normal/best.pt | 推理用 checkpoint |
| `SEG_MODEL` | models/seg_4_classes_all_0709_train_aug3/best.pt | YOLO 分割模型 |
| `OBB_MODEL` | models/inner_obb_clean_v11m_0129/best.pt | YOLO OBB 模型 |

## 前置数据

需要以下数据文件（已通过软链接配置）：
- `data/pcd_dataset_roi/` — 11135 个 .npz 点云文件（~5.1GB）
- `models/seg_4_classes_all_0709_train_aug3/best.pt` — YOLO 分割模型
- `models/inner_obb_clean_v11m_0129/best.pt` — YOLO OBB 模型
- `models/hf_cache/` — HuggingFace 缓存（MoGe 模型 ~1.2GB）

## 已知结果

| 模型 | 变体 | 验证 Mean Ang Err |
|---|---|---|
| NormalNet | Soft 课程 (纯 xyz) | 5.39° |
| MSECNet | soft, inlier≥0.8, agree≤15°, 1024pt | 待复现 |
