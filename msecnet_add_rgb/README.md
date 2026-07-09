# MSECNet + MoGe RGB Feature Fusion

本目录实现 MSECNet 点云几何特征与 MoGe/DINOv2 RGB 特征的融合，用于油箱盖法向量回归。

核心流程：

1. 用 MoGe v2 的 DINOv2 encoder 离线提取每张图像的 `feat_map` 和 `cls_token`
2. 训练时把 3D 点投影回图像，在 `feat_map` 上采样对应位置的视觉特征
3. 对 patch 内点特征做 `max pool + mean pool`，再拼接 `cls_token`
4. 通过门控融合到 MSECNet 几何特征，预测法向量

## 当前结果

当前完整融合实验目录：

```text
msecnet_add_rgb/ckpt
```

训练配置：

```text
labels: shared/normal_labels_patch03.npz
pcd_dir: data/pcd_dataset_roi
moge_feat_dir: data/moge_features
train / val: 9723 / 300
rgb_mode: full
rgb_feat_dim: 3072
steps: 12000
batch_size: 12
max_points: 0
aug_deg: 0
lr: 5e-4
```

最佳 checkpoint：

```text
msecnet_add_rgb/ckpt/best.pt
```

验证集结果：

| Step | Mean Angular Error | Median | <=10deg |
|------|--------------------|--------|---------|
| 1000 | 9.01 deg | 6.59 deg | 72.3% |
| 2000 | 6.44 deg | 6.02 deg | 83.7% |
| 3000 | 4.81 deg | 4.51 deg | 95.0% |
| 4000 | 4.60 deg | 4.13 deg | 96.0% |
| 5000 | 5.99 deg | 5.61 deg | 88.0% |
| 6000 | 4.91 deg | 4.58 deg | 95.7% |
| 7000 | 3.73 deg | 3.24 deg | 99.0% |
| 8000 | 3.91 deg | 3.54 deg | 98.3% |
| 9000 | 3.02 deg | 2.67 deg | 99.7% |
| 10000 | 2.76 deg | 2.44 deg | 99.3% |
| 11000 | 2.59 deg | 2.45 deg | 100.0% |
| 12000 | 2.38 deg | 2.25 deg | 100.0% |

最后一步验证集分布：

```text
mean: 2.38 deg
median: 2.25 deg
max: 8.45 deg
p75: 3.05 deg
p90: 3.97 deg
p95: 4.82 deg
<=3deg: 224 / 300
<=5deg: 288 / 300
<=10deg: 300 / 300
```

注意：这是当前数据 split 内验证结果。要确认泛化能力，需要和纯几何 baseline 使用同一 split 对比，并进一步做按车型/来源隔离的测试集。

## 环境

```bash
cd /data2/shendu/code/ruoyu/train_point2normal
conda activate point2normal
```

如果使用 Hugging Face 离线缓存，建议运行预计算时清掉代理变量：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    HF_HOME=/data2/shendu/code/ruoyu/fuelcap_6dpose/models/hf_cache \
    HF_HUB_OFFLINE=1 \
    python ...
```

## 预计算 MoGe 特征

训练前必须先生成 MoGe 特征：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    HF_HOME=/data2/shendu/code/ruoyu/fuelcap_6dpose/models/hf_cache \
    HF_HUB_OFFLINE=1 \
    python msecnet_add_rgb/precompute_moge_feat.py \
        data/pcd_dataset_roi \
        data/moge_features \
        --image-dir data/yolo_seg_by_car \
        --resolution-level 9 \
        --save-dtype float16 \
        --device cuda
```

输出文件：

```text
data/moge_features/<sample_name>.npz
```

每个文件包含：

```text
feat_map   # (1024, h_low, w_low), 低分辨率 MoGe/DINOv2 特征图
cls_token  # (1024,), 整张图的全局视觉特征
h, w       # 原始图像尺寸
```

当前数据里的 `pcd_dataset_roi/*.npz` 存的是点颜色 `(N,3)`，不是完整图像 `(H,W,3)`，所以需要 `--image-dir data/yolo_seg_by_car`。脚本会按 `车名__图片名.npz` 自动查找 `data/yolo_seg_by_car/车名/图片名.png`。

如果只想补算缺失或坏掉的特征文件，不要加 `--overwrite`。先移走坏文件，再运行预计算脚本，它会跳过已有正常文件。

## 训练 Full 模型

完整融合模式：

```text
rgb_mode=full = max_pool(feat_map) + mean_pool(feat_map) + cls_token
rgb_feat_dim = 1024 + 1024 + 1024 = 3072
```

训练命令：

```bash
python msecnet_add_rgb/train_rgb_fusion.py \
    shared/normal_labels_patch03.npz \
    data/pcd_dataset_roi \
    data/moge_features \
    --soft \
    --steps 12000 \
    --bs 12 \
    --max-points 0 \
    --aug-deg 0 \
    --rgb-mode full \
    --lr 5e-4 \
    --val-every 1000 \
    --out msecnet_add_rgb/ckpt
```

关键参数：

```text
--soft          使用 soft curriculum，对所有样本加权训练
--max-points 0  使用所有点，不做点云子采样
--aug-deg 0     RGB/MoGe 是相机视角特征，默认不要旋转点云
--rgb-mode      full / map / cls 三种 RGB 特征融合方式
--log-every     训练 loss/lr 写日志的间隔，默认 100
--vis-every     验证可视化间隔，默认 1000；设 0 可关闭
--vis-samples   每次可视化误差最大的验证样本数量，默认 6
```

训练输出都写到 `--out` 目录：

```text
best.pt
train.log
metrics.csv
metrics.jsonl
curves.png
run_config.json
val_predictions/latest.json
val_predictions/step_XXXXXX.json
val_vis/step_XXXXXX_error_summary.png
val_vis/step_XXXXXX/*_normal.png
val_vis/step_XXXXXX/*_featmap.png
```

其中：

```text
*_normal.png   3D点云法向对比，绿色是标签，红色是预测
*_featmap.png  MoGe feature map 的特征强度热力图
```

## 推理

本目录现在提供两种推理入口：

```text
infer.py           已有 MoGe 特征目录时，对 pcd_dir 批量推理
infer_pipeline.py  业务级完整 pipeline：RGB 图 + cloud.npz -> YOLO mask -> 在线/缓存 MoGe 特征 -> normal
```

### 已有 MoGe 特征批量推理

对已有点云数据和已预计算 MoGe 特征批量推理：

```bash
python msecnet_add_rgb/infer.py \
    msecnet_add_rgb/ckpt/best.pt \
    data/pcd_dataset_roi \
    data/moge_features \
    shared/msecnet_rgb_predictions.json
```

推理时必须提供对应的 MoGe 特征目录。`rgb_mode` 和 `rgb_feat_dim` 会优先从 checkpoint 读取，通常不用手动指定。

### 完整业务 Pipeline 推理

输入一张 RGB 图和一个点云 `.npz`，脚本内部会：

```text
1. YOLO seg 检测 inner_cover mask
2. 用 npz label 和 YOLO mask 过滤点云
3. 对 RGB 图在线提取 MoGe/DINOv2 feat_map + cls_token
4. 按训练时同样的规则构造 RGB descriptor
5. MSECNet + MoGe RGB fusion 输出法向量
```

命令：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    HF_HOME=/data2/shendu/code/ruoyu/fuelcap_6dpose/models/hf_cache \
    HF_HUB_OFFLINE=1 \
    python msecnet_add_rgb/infer_pipeline.py \
        msecnet_add_rgb/ckpt/best.pt \
        path/to/image.png \
        path/to/cloud.npz \
        output_normal.json \
        --moge-cache-dir data/moge_features_pipeline \
        --device cuda
```

默认 YOLO seg 模型路径：

```text
models/seg_4_classes_all_0709_train_aug3/best.pt
```

也可以显式指定：

```bash
python msecnet_add_rgb/infer_pipeline.py \
    msecnet_add_rgb/ckpt/best.pt \
    path/to/image.png \
    path/to/cloud.npz \
    output_normal.json \
    --seg-model models/seg_4_classes_all_0709_train_aug3/best.pt \
    --moge-cache-dir data/moge_features_pipeline \
    --device cuda
```

如果想沿用旧 pipeline 的 knob 局部 patch 裁剪，可以额外启用 OBB 模型：

```bash
python msecnet_add_rgb/infer_pipeline.py \
    msecnet_add_rgb/ckpt/best.pt \
    path/to/image.png \
    path/to/cloud.npz \
    output_normal.json \
    --use-obb \
    --obb-model models/inner_obb_clean_v11m_0129/best.pt \
    --moge-cache-dir data/moge_features_pipeline \
    --device cuda
```

调试时可以跳过 YOLO，只用 `.npz` 里的 `label` 字段：

```bash
python msecnet_add_rgb/infer_pipeline.py \
    msecnet_add_rgb/ckpt/best.pt \
    path/to/image.png \
    path/to/cloud.npz \
    output_normal.json \
    --no-yolo \
    --moge-cache-dir data/moge_features_pipeline \
    --device cuda
```

`--moge-cache-dir` 是可选的，但建议保留。第一次运行会在线提取 MoGe 特征并缓存为 `<cloud文件名>.npz`，后续同一张图会直接复用缓存，避免每次都加载 MoGe 大模型。

### Web 可视化推理结果

`web_label` 默认仍然是全量标注/查看队列：

```bash
python web_label/server.py --port 8765
```

如果只想查看 `infer_pipeline.py` 生成的单张 `output_normal.json`，传 `--pred-json`：

```bash
FULL_CLOUD=0 \
python web_label/server.py \
    --port 8766 \
    --pred-json output_normal.json \
    --src-dir /data2/shendu/code/ruoyu/fuelcap_6dpose/data/yolo_seg_by_car
```

然后打开：

```text
http://127.0.0.1:8766
```

说明：

```text
--pred-json output_normal.json  读取单张 pipeline 推理结果，并自动聚焦到对应 cloud_npz
--src-dir                       原图目录；当前数据通常在 fuelcap_6dpose/data/yolo_seg_by_car
FULL_CLOUD=0                    不等待 MoGe dense cloud worker，直接用 ROI 点云，打开更快
```

页面里的橙色箭头是 `output_normal.json` 中的预测法向；红色箭头是几何预标/可编辑法向。

如果端口被占用或想关闭服务：

```bash
lsof -ti:8766 | xargs -r kill
```

## 消融实验

消融目标是回答三个问题：

1. `cls_token` 是否单独有效
2. 投影到点云 patch 的 `feat_map` 是否单独有效
3. `full = map + cls` 是否比单分支更好

三组实验必须尽量保持其它参数一致：

```text
labels, pcd_dir, moge_feat_dir
--soft
--steps
--bs
--max-points
--aug-deg
--lr
--val-every
```

### 1. CLS-only

只使用整张图的 `cls_token`：

```bash
python msecnet_add_rgb/train_rgb_fusion.py \
    shared/normal_labels_patch03.npz \
    data/pcd_dataset_roi \
    data/moge_features \
    --soft \
    --steps 12000 \
    --bs 12 \
    --max-points 0 \
    --aug-deg 0 \
    --rgb-mode cls \
    --lr 5e-4 \
    --val-every 1000 \
    --out msecnet_add_rgb/ckpt_ablate_cls
```

### 2. Map-only

只使用投影到点云 patch 的 `feat_map` 池化特征：

```bash
python msecnet_add_rgb/train_rgb_fusion.py \
    shared/normal_labels_patch03.npz \
    data/pcd_dataset_roi \
    data/moge_features \
    --soft \
    --steps 12000 \
    --bs 12 \
    --max-points 0 \
    --aug-deg 0 \
    --rgb-mode map \
    --lr 5e-4 \
    --val-every 1000 \
    --out msecnet_add_rgb/ckpt_ablate_map
```

### 3. Full

使用 `feat_map` 池化特征和 `cls_token`：

```bash
python msecnet_add_rgb/train_rgb_fusion.py \
    shared/normal_labels_patch03.npz \
    data/pcd_dataset_roi \
    data/moge_features \
    --soft \
    --steps 12000 \
    --bs 12 \
    --max-points 0 \
    --aug-deg 0 \
    --rgb-mode full \
    --lr 5e-4 \
    --val-every 1000 \
    --out msecnet_add_rgb/ckpt_ablate_full
```

如果已经有 `msecnet_add_rgb/ckpt` 的 full 结果，可以先只跑 `cls` 和 `map`，再把已有 full 结果填入对比表。

### 并行跑消融

如果机器有多张空闲 GPU，可以用不同 `CUDA_VISIBLE_DEVICES` 并行跑：

```bash
CUDA_VISIBLE_DEVICES=0 python msecnet_add_rgb/train_rgb_fusion.py \
    shared/normal_labels_patch03.npz data/pcd_dataset_roi data/moge_features \
    --soft --steps 12000 --bs 12 --max-points 0 --aug-deg 0 \
    --rgb-mode cls --lr 5e-4 --val-every 1000 \
    --out msecnet_add_rgb/ckpt_ablate_cls &

CUDA_VISIBLE_DEVICES=1 python msecnet_add_rgb/train_rgb_fusion.py \
    shared/normal_labels_patch03.npz data/pcd_dataset_roi data/moge_features \
    --soft --steps 12000 --bs 12 --max-points 0 --aug-deg 0 \
    --rgb-mode map --lr 5e-4 --val-every 1000 \
    --out msecnet_add_rgb/ckpt_ablate_map &

wait
```

### 汇总消融结果

训练完成后读取每个目录的 `metrics.csv` 或 `best.pt`。最简单看 `metrics.csv` 最后一条验证记录：

```bash
for d in \
  msecnet_add_rgb/ckpt_ablate_cls \
  msecnet_add_rgb/ckpt_ablate_map \
  msecnet_add_rgb/ckpt_ablate_full \
  msecnet_add_rgb/ckpt
do
  echo "== $d =="
  tail -n 1 "$d/metrics.csv"
done
```

更稳妥的方式是扫描每个 `metrics.csv` 中最好的验证记录：

```bash
python - <<'PY'
import csv
import os

dirs = [
    ("CLS-only", "msecnet_add_rgb/ckpt_ablate_cls"),
    ("Map-only", "msecnet_add_rgb/ckpt_ablate_map"),
    ("Full-ablate", "msecnet_add_rgb/ckpt_ablate_full"),
    ("Full-current", "msecnet_add_rgb/ckpt"),
]

for name, d in dirs:
    path = os.path.join(d, "metrics.csv")
    if not os.path.exists(path):
        print(f"{name:12s} missing: {path}")
        continue
    with open(path, newline="", encoding="utf-8") as f:
        vals = [r for r in csv.DictReader(f) if r["split"] == "val"]
    best = min(vals, key=lambda r: float(r["mean_ang_err"]))
    print(
        f"{name:12s} step={best['step']:>5s} "
        f"mean={float(best['mean_ang_err']):.3f} "
        f"median={float(best['median_ang_err']):.3f} "
        f"p10={float(best['p10']):.1f}"
    )
PY
```

建议整理成表：

| Experiment | rgb_mode | rgb_feat_dim | Mean Angular Error | Median | <=10deg | Best Step |
|------------|----------|--------------|--------------------|--------|---------|-----------|
| Geometry baseline | none | 0 | 6.17 deg | - | - | - |
| CLS-only | cls | 1024 | 待填 | 待填 | 待填 | 待填 |
| Map-only | map | 2048 | 待填 | 待填 | 待填 | 待填 |
| Full | full | 3072 | 2.38 deg | 2.25 deg | 100.0% | 12000 |

判断方式：

```text
cls 好，map 一般：全局语义/车型先验贡献更大
map 好，cls 一般：目标区域局部视觉几何贡献更大
full 最好：全局语义和局部特征互补
full 不如单分支：融合头或特征冗余需要调整
```

### Augmentation 消融

RGB/MoGe 特征固定在相机视角，旋转点云会破坏 RGB 与几何的对齐。默认使用：

```text
--aug-deg 0
```

如果要确认这个假设，可以额外跑：

```bash
python msecnet_add_rgb/train_rgb_fusion.py \
    shared/normal_labels_patch03.npz \
    data/pcd_dataset_roi \
    data/moge_features \
    --soft \
    --steps 12000 \
    --bs 12 \
    --max-points 0 \
    --aug-deg 15 \
    --rgb-mode full \
    --lr 5e-4 \
    --val-every 1000 \
    --out msecnet_add_rgb/ckpt_ablate_full_aug15
```

如果 `aug15` 明显变差，说明 RGB 与点云投影对齐确实重要。

## 实现细节

### 特征投影

```python
u = K_norm[0, 0] * w * x / z + K_norm[0, 2] * w
v = K_norm[1, 1] * h * y / z + K_norm[1, 2] * h
point_feats = bilinear_sample(feat_map, u, v)
```

`K_norm` 是归一化相机内参，先乘 `w/h` 还原像素坐标，再映射到低分辨率特征图。

### RGB 全局特征

```python
rgb_global = concat([
    max_pool(point_feats),  # (1024,)
    mean_pool(point_feats), # (1024,)
    cls_token               # (1024,)
])
```

不同 `rgb_mode` 的维度：

```text
cls:  1024
map:  2048
full: 3072
```

### 门控融合

```python
rgb_feat = rgb_proj(rgb_global)
alpha = sigmoid(Linear(geom_feat))
fused = concat(geom_feat, alpha * rgb_feat_per_sample)
normal = fusion_classifier(fused)
```

Loss 仍然是法向量角度相关损失：

```python
loss = 1 - (pred_normal · target_normal) ** 2
```

加 RGB 只改变输入和融合方式，不改变监督目标。

## 故障排查

### `EOFError: No data left in file`

说明某个 MoGe 特征 `.npz` 是空文件或损坏文件。检查：

```bash
find data/moge_features -maxdepth 1 -type f -name '*.npz' -size 0 -print
```

处理方式：

```bash
mkdir -p data/moge_features_bad_empty
find data/moge_features -maxdepth 1 -type f -name '*.npz' -size 0 \
    -exec mv -t data/moge_features_bad_empty {} +
```

然后重新运行预计算脚本补齐缺失文件。

### RGB 特征维度不匹配

训练脚本会从 `moge_feat_dir/*.npz` 自动推导 `feat_map` 和 `cls_token` 维度，并把 `rgb_feat_dim` / `rgb_mode` 写入 checkpoint。推理时应使用同一套 `data/moge_features`。

### 显存不足

减小 batch size：

```bash
python msecnet_add_rgb/train_rgb_fusion.py ... --bs 6
```

或启用点云子采样：

```bash
python msecnet_add_rgb/train_rgb_fusion.py ... --max-points 512
```

### Matplotlib 不可用

训练本身不依赖可视化。如果环境没有 `matplotlib`，脚本会跳过 `curves.png` 和 `val_vis` 图片，但仍然保存 `train.log`、`metrics.csv` 和 `val_predictions/*.json`。
