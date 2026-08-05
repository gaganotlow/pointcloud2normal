# point2normal 使用手册

本仓库有三个**并列**的训练任务。它们都只使用最新的
`data/fuelcap_pass_20260803_10847/`，各自训练、评估并保存 checkpoint；没有哪个任务是“附属功能”。

| 任务 | 目录 | 要解决的问题 | 输入 | 评价 |
| --- | --- | --- | --- | --- |
| 8 cm 球形点云 | `msecnet_ball/` | 人工指定盖中心的朝相机法向 | 中心周围 8 cm 球形点云 | 有向 `angular_error_deg` |
| RGB + 球形点云 | `msecnet_ball_addRGB/` | 同一中心法向，结合原图视觉信息 | 8 cm 球形点云 + OBB RGB crop | 有向 `angular_error_deg` |
| 人工伪 OBB | `msecnet_best/` | 历史 MSECNet 局部平面法向基线 | 人工矩形定义的薄点云棱柱 | 无向 `axis_error_deg` |

球形与 RGB 任务的法向方向必须朝相机，预测反向即错。伪 OBB 的正反向等价，因此它的误差不能与前两个任务直接比较。

## 1. 环境

所有命令均在仓库根目录、`point2normal` Conda 环境中执行：

```bash
cd /data2/shendu/code/ruoyu/train_point2normal
conda run --no-capture-output -n point2normal python <脚本> <参数>
```

三个任务共用 `msecnet_best/MSECNet/` 中的 pointops CUDA 扩展。首次配置环境时编译一次：

```bash
(
  cd msecnet_best/MSECNet/scripts/lib/pointops
  CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    conda run --no-capture-output -n point2normal python setup.py install
)
```

## 2. 唯一训练数据

当前唯一允许用于训练、微调和 RGB 融合的原始集是：

```text
data/fuelcap_pass_20260803_10847/
```

它包含 10,847 个审核通过样本、145 个车型。旧 9211 及更早数据集只保留作历史追溯，不能参与当前训练。三个训练脚本会检查派生集 `dataset.json` 的 `source_dataset`，并拒绝旧集或混合的 labels、clouds、anchors、split 路径。

| 派生集 | 样本与切分 | 供哪个任务使用 |
| --- | --- | --- |
| `msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08` | 10,846；train 8,676 / val 1,085 / test 1,085 | 8 cm 球形点云、RGB + 球形点云 |
| `msecnet_best_fuelcap_pass_20260803_10847_manual3d_pseudo_obb` | 10,826；train 8,635 / val 1,103 / test 1,088 | 人工伪 OBB |

每个派生集的 `manifest.jsonl` 用于回溯原始样本，`split_*.json` 是唯一有效的 train/val/test 划分。不要随机重切帧，也不要覆盖当前目录；修改裁剪参数时创建新的输出目录，并重新生成 `dataset.json`。

更完整的数据字段、坐标系和数据集关系见 [data/readme.md](data/readme.md)。

## 3. 任务一：8 cm 球形点云

**目标：** 回归人工 `center_3d` 处、朝相机的盖面法向。

**输入：** 从原始全量点云取 `||point - center_3d|| <= 0.08 m` 的球。球内的边缘、孔洞和背景点都是上下文；标签仍是球心处的人工 `normal`。训练输入会转为 `(point - center_3d) / 0.08`，并使用半径特征和径向权重。

### 准备

当前派生集已经生成。需要以相同协议重新制备时：

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball/prepare_ball_dataset.py \
  data/fuelcap_pass_20260803_10847 \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08 \
  --ball-radius-m 0.08 --max-points 4096
```

### 训练

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball/train.py \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/clouds \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/split_by_generalization_group.json \
  --ball-radius-m 0.08 --radial-weight-beta 2.0 \
  --steps 70000 --bs 24 --max-points 1024 --aug-deg 45 \
  --point-loss-weight 0.25 --seed 20260722 \
  --early-stop-patience 100 --snapshot-every 1000 \
  --out msecnet_ball/out/center_ball_r08_oriented_20260803_v1
```

### 评估

```bash
conda run --no-capture-output -n point2normal python msecnet_ball/infer.py \
  msecnet_ball/out/center_ball_r08_oriented_20260803_v1/best.pt \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/clouds \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/split_by_generalization_group.json \
  --split-name test
```

产物位于 `msecnet_ball/out/center_ball_r08_oriented_20260803_v1/`：`best.pt`、`last.pt`、`metrics.csv`、`dashboard.png`、`run.json` 和 `inference_test/report.json`。

## 4. 任务二：RGB + 球形点云

**目标：** 回归与任务一相同的有向法向，同时让原图 RGB 为点云几何提供补充信息。

**输入：** 与任务一完全相同的 8 cm 球形点云，加上源 RGB 图中检测器 OBB 定位的相机朝向 crop。该任务保留并冻结一个已验证的 10847 球形点云 checkpoint，RGB 分支以受限残差修正几何预测；这是本任务的模型设计，不改变它作为独立训练与评估任务的地位。

### 准备

先为 10847 的有标签样本缓存 OBB 检测结果。此步骤可中断后重跑，已完成的样本会复用：

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball_addRGB/detect_obb.py \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/labels_manual3d.npz \
  --source-root data/fuelcap_pass_20260803_10847 \
  --obb-model ../fuelcap_6dpose/models/inner_obb_clean_v11m_0129/best.pt \
  --out data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/obb_inner_0129.json \
  --class-id 0 --conf 0.25 --batch-size 32
```

### 训练

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball_addRGB/train.py \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/clouds \
  --source-root data/fuelcap_pass_20260803_10847 \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/split_by_generalization_group.json \
  --obb-detections data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/obb_inner_0129.json \
  --obb-crop-scale 3.5 --image-backbone dino_vits14 --image-size 336 \
  --dino-unfreeze-blocks 1 --dino-lr 3e-6 \
  --geometry-mode pretrained_point \
  --geometry-checkpoint msecnet_ball/out/center_ball_r08_oriented_20260803_v1/best.pt \
  --freeze-geometry --fusion-mode point_aligned_residual \
  --max-rgb-correction 0.05 --initial-gate 0.10 --gate-penalty 0.001 \
  --point-loss-weight 0 --geometry-loss-weight 0 --image-loss-weight 0.05 \
  --rgb-dropout 0.00 --steps 30000 --batch-size 16 --max-points 1024 \
  --out msecnet_ball_addRGB/out/rgb_fusion_dino_obb_20260803_v1
```

### 评估

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball_addRGB/infer.py \
  msecnet_ball_addRGB/out/rgb_fusion_dino_obb_20260803_v1/best.pt \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/clouds \
  --source-root data/fuelcap_pass_20260803_10847 \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/split_by_generalization_group.json \
  --obb-detections data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/obb_inner_0129.json \
  --split-name test
```

产物位于 `msecnet_ball_addRGB/out/rgb_fusion_dino_obb_20260803_v1/`：`best.pt`、`last.pt`、`metrics.csv`、`dashboard.png`、`run.json` 和 `inference_test/report.json`。评估时同时查看融合、纯几何和纯 RGB 的预测，不要只看融合均值。

## 5. 任务三：人工伪 OBB

**目标：** 复现并维护历史 MSECNet 局部平面基线，回归法向轴而非有方向的法向量。

**输入：** 围绕人工 `center_3d` 和 `pose_T` 的薄长方体：面内为人工 `disc_wh` 的 2 倍，法向半厚为 5 mm。人工矩形只决定输入区域；监督标签仍为人工 `normal`。

### 准备

当前派生集已经生成。需要以相同协议重新制备时：

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_best/prepare_pseudo_obb_dataset.py \
  data/fuelcap_pass_20260803_10847 \
  data/msecnet_best_fuelcap_pass_20260803_10847_manual3d_pseudo_obb \
  --obb-expand 2.0 --obb-half-depth-m 0.005 --max-points 4096
```

### 训练

```bash
conda run --no-capture-output -n point2normal python msecnet_best/train.py \
  data/msecnet_best_fuelcap_pass_20260803_10847_manual3d_pseudo_obb/labels_manual3d.npz \
  data/msecnet_best_fuelcap_pass_20260803_10847_manual3d_pseudo_obb/clouds \
  --centers data/msecnet_best_fuelcap_pass_20260803_10847_manual3d_pseudo_obb/anchors_manual3d.json \
  --split data/msecnet_best_fuelcap_pass_20260803_10847_manual3d_pseudo_obb/split_by_car_model.json \
  --steps 70000 --bs 24 --max-points 1024 --radius 0.3 --aug-deg 45 \
  --snapshot-every 1000 \
  --out msecnet_best/out/pseudo_obb_20260803_v1
```

### 评估

```bash
conda run --no-capture-output -n point2normal python msecnet_best/infer.py \
  msecnet_best/out/pseudo_obb_20260803_v1/best.pt \
  data/msecnet_best_fuelcap_pass_20260803_10847_manual3d_pseudo_obb/labels_manual3d.npz \
  data/msecnet_best_fuelcap_pass_20260803_10847_manual3d_pseudo_obb/clouds \
  --centers data/msecnet_best_fuelcap_pass_20260803_10847_manual3d_pseudo_obb/anchors_manual3d.json \
  --split data/msecnet_best_fuelcap_pass_20260803_10847_manual3d_pseudo_obb/split_by_car_model.json \
  --split-name test
```

产物位于 `msecnet_best/out/pseudo_obb_20260803_v1/`：`best.pt`、`last.pt`、`snapshots/`、`metrics.csv`、`dashboard.png`、`run.json` 和 `inference_test/report.json`。

## 6. 共享工具

### 开放集预测

`data/msecnet_20260730_open_obb10_unlabeled_test/` 有 42 个独立开放集样本，没有人工法向标签。它仅用于观察预测和一致性，不能用于训练、模型选择或角度误差计算。

### Web 查看

```bash
bash shell/launch_msecnet_test_viewer.sh 8765
```

打开 `http://127.0.0.1:8765` 后，选择已完成 checkpoint、匹配的派生集和 split。Web UI 支持伪 OBB 与单模态球形任务；RGB 融合请使用其 `infer.py` 生成报告。

### 法向标签修复 Web

对 `train` 或 `val` 做权重推理后，可以在同一页面直接复核并调整人工法向：

```bash
bash shell/launch_fix_normal.sh 8766
```

打开 `http://127.0.0.1:8766`，选择权重、匹配的准备数据集和 `train`/`val`，再点“推理并载入”。默认只排入“预测与当前标签的误差**大于** 5°”的样本；取消筛选或修改阈值后无需重复推理。红箭头是可编辑的当前标签，橙箭头是预测，青箭头是原始标签；拖动红色旋转环或用方向键微调，按空格保存并进入下一条。左侧“显示：模型输入”可切换到该样本的全局原始点云，右上角同步显示源 RGB 图。

保存会在一次操作中同步更新所选派生训练集中的 `labels_manual3d.npz` 与 `anchors_manual3d.json`：后者的 `normal`、`tangent` 和 `pose_T` 会重建为同一个正交坐标系，避免两份标签再次漂移。训练和后续推理会立刻使用新标签；每个文件均以原子替换写入。修复应用还会更新：

- `normal_fixes.json`：记录原始/上一次/修复后法向、推理预测、当时误差、权重、报告、划分和保存时间。

页面重新推理和三个训练脚本都直接读取这个被修复后的 `labels_manual3d.npz`，并仍只接受 20260803 的 clouds、anchors、split 与原始数据源。

### PLY 与 ONNX

- `msecnet_ball/export_pseudo_obb_ply.py`：导出球形训练点、原始上下文和人工标签。
- `msecnet_best/export_pseudo_obb_ply.py`：导出伪 OBB 点和原始上下文。
- `msecnet_best/export_onnx.py`：仅导出伪 OBB 历史模型；球形和 RGB 任务没有 ONNX 导出器。

各任务目录 README 记录各自的模型结构与专用参数；本文件负责保证三条训练路线的地位、数据源和基本工作流保持对称。
