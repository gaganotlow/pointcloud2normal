# point2normal 使用手册

本仓库用于从油箱内盖附近的点云回归相机坐标系下的三维法向量。当前包含三个 MSECNet 实验：历史伪 OBB 基线、以人工中心为球心的 8 cm 球形点云，以及 RGB 与球形点云的晚融合实验。

所有命令均从仓库根目录执行：

```bash
cd /data2/shendu/code/ruoyu/train_point2normal
conda run --no-capture-output -n point2normal python <脚本> <参数>
```

## 1. 当前路线与适用范围

| 路线 | 目录 | 输入 | 标签方向/指标 | 当前状态 |
| --- | --- | --- | --- | --- |
| 历史伪 OBB | `msecnet_best/` | 人工伪 OBB 局部棱柱点云 | 无向，`abs(dot)` 轴向误差 | 可用的历史基线 |
| 8 cm 球 | `msecnet_ball/` | 人工 3D 中心周围 0.08 m 球 | 朝相机的有向法向，角度误差 | 可训练、可评估 |
| RGB 融合球 | `msecnet_ball_addRGB/` | 8 cm 球 + 源图 RGB crop | 朝相机的有向法向，角度误差 | 当前可训练实验 |

三条路线的 checkpoint 互不兼容，也不能直接比较无向轴向误差与有向角度误差。

## 2. 环境与 CUDA pointops

`environment.yml` 固定 Python 3.11、PyTorch 2.6 CUDA 12.4、OpenCV、Flask、Ultralytics 和 MoGe。首次在新环境使用 MSECNet 前，需要编译 `pointops` CUDA 扩展。当前可用副本位于 `msecnet_best/MSECNet/`：

```bash
(
  cd msecnet_best/MSECNet/scripts/lib/pointops
  CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    conda run --no-capture-output -n point2normal python setup.py install
)
```

用以下命令确认 GPU：

```bash
conda run --no-capture-output -n point2normal python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3. 数据集

### 3.1 源数据集：9211 帧

`data/fuelcap_pass_20260721_9211/` 是三个有标签实验的共同源数据。`index.jsonl` 每行描述一个样本，关联 `cloud_full`、`cloud_local`、`image`、`label`、车型和来源数据集。

源云 `.npz` 的主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `xyz (N,3)` | CV 相机坐标系中的米制三维点 |
| `rgb (N,3)` | 与源云生成过程关联的颜色；当前球形点云训练不逐点使用它 |
| `normal (N,3)` | 逐点表面法向，不能与监督标签混用 |
| `label (N,)` | `1` 为 3D 内盖分割 |
| `K_norm`, `w`, `h` | 归一化相机内参与图像尺寸 |

人工标签 `labels/<样本>.json` 中的 `normal`、`center`、`tangent`、`pose_T`、`disc_wh` 均在同一相机坐标系。所有训练标签法向均朝相机，即通常满足 `normal dot (-center) > 0`。

### 3.2 已准备训练集

| 数据集 | 样本/划分 | 输入选择 | 对应模型 |
| --- | --- | --- | --- |
| `data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/` | 9189；train 7349 / val 921 / test 919 | 人工矩形定义的薄伪 OBB 棱柱 | `msecnet_best` |
| `data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/` | 9211；train 7369 / val 921 / test 921 | 人工 3D 中心半径 0.08 m 的球 | `msecnet_ball`、`msecnet_ball_addRGB` |
| `data/msecnet_20260730_open_obb10_unlabeled_test/` | 42；仅 test | 检测 OBB 投影到点云后的裁剪 | 无标签推理 |

有标签准备集的共同文件：

```text
clouds/<样本>.npz              已准备的局部点云
labels_manual3d.npz            files、normal、inlier_frac、agree_deg、n_inner
anchors_manual3d.json          人工中心、姿态、半径或伪 OBB 来源
manifest.jsonl                 源云、车型、划分及可追溯信息
split_*.json                   不泄漏的 train / val / test 划分
```

不要修改已经生成的 `clouds/`、`labels_manual3d.npz` 或 split 文件。需要变更半径、OBB 参数或筛选规则时，请使用新的输出目录重新准备数据。

## 4. 数据准备

### 4.1 伪 OBB 数据集

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_best/prepare_pseudo_obb_dataset.py \
  data/fuelcap_pass_20260721_9211 \
  data/new_pseudo_obb_dataset \
  --obb-expand 2.0 --obb-half-depth-m 0.005 --max-points 4096
```

该脚本按车型分离 train/val/test。`--resume` 仅用于中断后的未完成输出目录；完整数据集不能覆盖。

### 4.2 8 cm 球形数据集

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball/prepare_ball_dataset.py \
  data/fuelcap_pass_20260721_9211 \
  data/new_ball_r08_dataset \
  --ball-radius-m 0.08 --max-points 4096
```

球心是人工审核的 `center_3d`，并非检测器 OBB 中心。点云坐标在训练中转换为 `(point - center_3d) / ball_radius_m`。

### 4.3 无标签开放集

开放集源目录需含每帧的 `input_color.jpg`、`camera_intrinsics.json` 和 `scene_pointcloud.ply`。已有固定配置可直接执行：

```bash
bash shell/prepare_20260730_open_obb10_test.sh
```

或调用 `shell/prepare_20260730_open_obb10_test.py SOURCE_DIR OUT_DIR --obb-model MODEL.pt`。输出只含预测输入和 OBB 元数据，没有人工法向标签，不能调用有标签评估脚本。

## 5. 训练与评估

### 5.1 历史伪 OBB 基线

```bash
conda run --no-capture-output -n point2normal python msecnet_best/train.py \
  data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/labels_manual3d.npz \
  data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/clouds \
  --centers data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/anchors_manual3d.json \
  --split data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/split_by_car_model.json \
  --steps 70000 --bs 24 --max-points 1024 --radius 0.3 --aug-deg 45 \
  --out msecnet_best/out/pseudo_obb_9211_v1

conda run --no-capture-output -n point2normal python msecnet_best/infer.py \
  msecnet_best/out/pseudo_obb_9211_v1/best.pt \
  data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/labels_manual3d.npz \
  data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/clouds \
  --centers data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/anchors_manual3d.json \
  --split data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/split_by_car_model.json \
  --split-name test
```

输出包含 `best.pt`、`last.pt`、`snapshots/`、`metrics.csv`、`dashboard.png` 和 `run.json`。报告中的 `axis_error_deg` 使用无向轴向误差。

### 5.2 RGB + 球形点云晚融合实验

```bash
conda run --no-capture-output -n point2normal python msecnet_ball_addRGB/train.py \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/clouds \
  --source-root data/fuelcap_pass_20260721_9211 \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/split_by_generalization_group.json \
  --steps 30000 --batch-size 16 --max-points 1024 \
  --rgb-dropout 0.20 --out msecnet_ball_addRGB/out/rgb_fusion_v1

conda run --no-capture-output -n point2normal python msecnet_ball_addRGB/infer.py \
  msecnet_ball_addRGB/out/rgb_fusion_v1/best.pt \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/clouds \
  --source-root data/fuelcap_pass_20260721_9211 \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/split_by_generalization_group.json \
  --split-name test
```

RGB 不与 3D 点逐点绑定：RGB 分支从人工 3D 中心投影处提取自适应图像 crop，点云分支保留 MSECNet 和径向加权池化，两个全局特征在末端融合。默认 `--aug-deg 0`，因为任意三维旋转无法同步变换相机图像。评估时应同时检查融合、纯几何与纯 RGB 预测，不要只看融合指标。

### 5.3 8 cm 点云单模态实验

`msecnet_ball/` 的训练、评估、无标签推理与球形 PLY 导出命令见该目录的 README。它使用半径特征 `r` 和权重 `exp(-beta*r^2)`，输出有向角度误差，并复用 `msecnet_best/MSECNet/` 的 MSECNet/pointops 实现。它的额外半径输入使 checkpoint 与伪 OBB 基线不兼容，必须从头训练。

## 6. 无标签预测、PLY 与 ONNX

历史伪 OBB 模型可对开放集产生法向预测：

```bash
conda run --no-capture-output -n point2normal python msecnet_best/predict_unlabeled.py \
  msecnet_best/out/pseudo_obb_9211_v1/best.pt \
  data/msecnet_20260730_open_obb10_unlabeled_test
```

导出一个伪 OBB 样本为彩色 PLY：

```bash
conda run --no-capture-output -n point2normal python msecnet_best/export_pseudo_obb_ply.py \
  data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb \
  --source-root data/fuelcap_pass_20260721_9211 --raw-points 50000
```

历史模型可导出为标准 ONNX。需要额外安装 `onnx`；输入固定为预处理后的 `points[1, N, 3]`，`N` 必须不少于 128 且能被 8 整除：

```bash
conda run --no-capture-output -n point2normal python msecnet_best/export_onnx.py \
  msecnet_best/out/pseudo_obb_9211_v1/best.pt \
  --out msecnet_best/out/pseudo_obb_9211_v1/best.onnx --num-points 1024
```

ONNX 仅覆盖历史伪 OBB 模型；球形和 RGB 融合模型没有对应导出器。

## 7. Web 可视化

### 7.1 交互式推理界面

```bash
bash shell/launch_msecnet_test_viewer.sh 8765
```

浏览器打开 `http://127.0.0.1:8765`。界面会扫描 `msecnet_best/out/*/best.pt` 和 `msecnet_ball/out/*/best.pt`，选择兼容数据集与 `train`、`val`、`test` 后启动一次 GPU 推理任务；结果写入 checkpoint 相邻的 `web_inference/<dataset>/<split>/`。目前 UI 不扫描 RGB 融合 checkpoint。

### 7.2 查看已完成报告

对有 `report.json` 的已有结果，可启动只读模式：

```bash
conda run --no-capture-output -n point2normal python web_label/server.py \
  --msecnet-report msecnet_best/out/pseudo_obb_9211_v1/inference_test/report.json \
  --msecnet-dataset data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb \
  --msecnet-source-root data/fuelcap_pass_20260721_9211 \
  --port 8766
```

只读模式显示准备点云、源点云上下文、人工目标和模型预测。对于无标签报告，不显示角度误差。普通标注模式会写入 `output/`，与评估只读模式不同。

## 8. 工具脚本索引

| 路径 | 作用 |
| --- | --- |
| `msecnet_best/prepare_pseudo_obb_dataset.py` | 用人工矩形生成伪 OBB 训练集 |
| `msecnet_best/train.py`、`infer.py` | 历史无向模型的训练与有标签评估 |
| `msecnet_best/predict_unlabeled.py` | 历史模型的无标签 OBB 推理 |
| `msecnet_best/export_pseudo_obb_ply.py` | 导出伪 OBB 与源云上下文 PLY |
| `msecnet_best/export_onnx.py` | 导出固定点数、标准算子 ONNX |
| `msecnet_ball/prepare_ball_dataset.py` | 从人工中心生成球形训练集 |
| `msecnet_ball/export_pseudo_obb_ply.py` | 批量导出球形训练样本 PLY |
| `msecnet_ball_addRGB/train.py`、`infer.py` | 独立 RGB/点云晚融合训练和评估 |
| `shell/prepare_20260730_open_obb10_test.py` | 检测 OBB 并生成无标签开放集 |
| `shell/launch_msecnet_test_viewer.sh` | 启动 Web 推理/浏览界面 |
| `web_label/server.py` | Flask 标注器、只读报告浏览和 Web 推理服务 |
| `web_label/moge_worker.py` | 可选常驻 MoGe worker，为标注器生成缓存源云 |
| `shared/cap_patch.py` | 历史局部盖面 patch 几何函数 |
| `shared/make_normal_labels.py` | 历史弱标签生成脚本；不支持 `--help`，必须带数据目录 |
| `shared/knob_centers_all.py` | 历史批量 OBB 中心生成脚本；需明确给出数据、图像和检测模型路径 |
| `shared/precompute_moge_normal.py` | 历史 MoGe 逐点法向聚合；依赖旧的 shared 产物 |

`shared/` 工具依赖历史的 `pcd_dataset_roi`、检测中心和弱标签文件，不是当前 9211 人工标注数据集的前置步骤。对当前训练优先使用第 4 节的准备脚本。

## 9. 常见问题与边界

- **点云 CUDA 扩展报错**：确认在 `point2normal` 环境中重新编译 `msecnet_best/MSECNet/scripts/lib/pointops`，并检查 PyTorch/CUDA ABI 是否改变。
- **准备数据时报输出目录已存在**：这是保护已有数据的设计。使用新的 `out_dir`；不要通过删除现有数据集来重跑。
- **有向/无向指标不一致**：`msecnet_best` 使用 `axis_error_deg`，翻转法向等价；球形和 RGB 融合使用有向 `angular_error_deg`，翻转即为错误。
- **RGB 融合效果不稳定**：检查 `source-root/index.jsonl` 到图像的映射、保持车型/采集域隔离的 test split，并比较几何分支、RGB 分支和融合分支。
- **开放集没有误差**：42 个开放集样本没有人工法向标签，只能查看预测和一致性，不能计算测试角度。
- **Web 页面没有 RGB checkpoint**：当前 Web UI 仅支持历史伪 OBB 与单模态球形协议；RGB 融合请先用其 `infer.py` 输出报告。
