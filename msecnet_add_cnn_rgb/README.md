# MSECNet + CNN/ResNet RGB

这个目录是 `msecnet_add_rgb` 的端到端 RGB 版本：不用 MoGe / DINOv2，不需要 `data/moge_features` 特征目录，训练和推理都直接输入 RGB 图像与点云。

当前支持两类 RGB encoder：

```text
light      手写 5 层轻量 CNN，速度快，已有 best mean_ang_err=3.16 deg
resnet*    torchvision ResNet18/34/50/101，可加载 ImageNet 预训练权重，默认 resnet50
```

目标是更贴近边缘端部署：

```text
RGB image + point cloud npz -> normal
```

而不是 MoGe 版本的两段式：

```text
RGB image -> MoGe/DINOv2 feature npz
feature npz + point cloud npz -> normal
```

## 设计

核心结构：

1. CNN/ResNet 从 RGB 图像提取低分辨率 feature map
2. 3D 点云 patch 通过 `K_norm` 投影回图像坐标
3. 在 RGB feature map 上用 bilinear sampling 取每个点的 RGB 特征
4. 对 patch 内 RGB 点特征做 `max pool + mean pool`
5. 与 MSECNet 几何特征做门控融合
6. 输出法向量

默认维度：

```text
RGB feature map channels: 128
RGB global feature: max_pool(128) + mean_pool(128) -> Linear -> 512
```

ResNet 默认取 `layer2` feature map。输入 `256x256` 时分辨率约为 `32x32`，更适合 3D 点投影后的 bilinear sampling；`layer3/layer4` 语义更强但空间更粗，容易损失局部对齐。

这个版本没有 MoGe 的大模型语义先验，精度未必能达到 `msecnet_add_rgb` 的 `2.38 deg`，但部署链路明显简单很多。

## 文件

```text
msecnet_add_cnn_rgb/
├── train_cnn_rgb.py   # 端到端训练：RGB + 点云 -> normal
├── infer_e2e.py       # 端到端推理：单张RGB + 单个点云npz -> normal.json
├── infer_pipeline.py  # YOLO + CNN/ResNet-RGB业务推理：自动找inner_cover/knob
├── run.sh             # 默认训练入口
├── ckpt_resnet50_001/ # run.sh 自动编号输出目录
├── ckpt/              # 已有 light CNN 训练结果
└── README.md
```

## 训练

```bash
cd /data2/shendu/code/ruoyu/train_point2normal
conda activate point2normal
```

推荐先跑 ResNet50 预训练配置：

```bash
python msecnet_add_cnn_rgb/train_cnn_rgb.py \
    shared/normal_labels_patch03.npz \
    data/pcd_dataset_roi \
    --image-dir data/yolo_seg_by_car \
    --soft \
    --steps 30000 \
    --bs 8 \
    --max-points 0 \
    --image-size 256 \
    --cnn-dim 128 \
    --rgb-feat-dim 512 \
    --rgb-backbone resnet50 \
    --resnet-stage layer2 \
    --freeze-rgb-backbone-steps 5000 \
    --rgb-backbone-lr-mult 0.05 \
    --aug-deg 0 \
    --lr 2e-4 \
    --val-every 1000
```

也可以直接用：

```bash
bash msecnet_add_cnn_rgb/run.sh
```

`run.sh` 默认等价于：

```text
RGB_BACKBONE=resnet50
RESNET_STAGE=layer2
RGB_PRETRAINED=1
FREEZE_RGB_BACKBONE_STEPS=5000
RGB_BACKBONE_LR_MULT=0.05
OUTPUT_DIR_PREFIX=msecnet_add_cnn_rgb/ckpt_resnet50
```

如果不显式传 `OUTPUT_DIR`，`run.sh` 会自动选择下一个不存在的目录。直接运行 `train_cnn_rgb.py` 且不传 `--out` 时也使用同样的编号规则：

```text
ckpt_resnet50_001
ckpt_resnet50_002
ckpt_resnet50_003
```

如果显存不够，可以退回 ResNet34：

```bash
RGB_BACKBONE=resnet34 BS=12 OUTPUT_DIR_PREFIX=msecnet_add_cnn_rgb/ckpt_resnet34 \
    bash msecnet_add_cnn_rgb/run.sh
```

如果要复现旧的手写 light CNN：

```bash
RGB_BACKBONE=light OUTPUT_DIR_PREFIX=msecnet_add_cnn_rgb/ckpt_light \
    bash msecnet_add_cnn_rgb/run.sh
```

训练输出：

```text
msecnet_add_cnn_rgb/ckpt_resnet50_001/
├── best.pt
├── train.log
├── metrics.csv
├── metrics.jsonl
├── curves.png
├── run_config.json
└── val_predictions/
```

关键参数：

```text
--image-dir       原始 RGB 图像目录
--image-size      CNN 输入尺寸，默认 256
--cnn-dim         CNN feature map 通道数，默认 128
--rgb-feat-dim    融合前 RGB 全局特征维度，默认 512
--rgb-backbone    light/resnet18/resnet34/resnet50/resnet101，默认 resnet50
--resnet-stage    layer2/layer3/layer4，默认 layer2
--no-rgb-pretrained  不加载 ImageNet 预训练权重
--freeze-rgb-backbone-steps  前 N 步冻结 ResNet trunk，只训练投影/融合/几何分支
--rgb-backbone-lr-mult  ResNet trunk 学习率倍率，默认 0.05
--no-freeze-rgb-bn  允许 ResNet BatchNorm 更新；默认冻结 BN
--max-points 0    使用所有点
--aug-deg 0       默认不旋转点云，避免破坏图像投影对应关系
```

## 推理

本目录提供两个推理入口：

```text
infer_e2e.py       模型级推理：RGB + cloud.npz，可选手动传knob center
infer_pipeline.py  业务级推理：RGB + cloud.npz，自动跑YOLO分割找inner_cover
```

实际部署优先使用 `infer_pipeline.py`。默认只跑一个 YOLO 分割模型；如果确实需要兼容旧 pipeline 的 knob 局部裁剪，再额外打开 `--use-obb`。

### 业务级推理：YOLO Seg

输入一张 RGB 图和一个点云 `.npz`，脚本内部会：

```text
1. YOLO seg 检测 inner_cover mask
2. 用 npz label 和 YOLO mask 过滤点云
3. CNN-RGB + MSECNet 输出内盖法向量
```

命令：

```bash
python msecnet_add_cnn_rgb/infer_pipeline.py \
    msecnet_add_cnn_rgb/ckpt_resnet50_001/best.pt \
    path/to/image.png \
    path/to/cloud.npz \
    output_normal.json \
    --device cuda
```

默认 YOLO 模型路径：

```text
SEG_MODEL: models/seg_4_classes_all_0709_train_aug3/best.pt
```

也可以显式传入：

```bash
python msecnet_add_cnn_rgb/infer_pipeline.py \
    msecnet_add_cnn_rgb/ckpt_resnet50_001/best.pt \
    path/to/image.png \
    path/to/cloud.npz \
    output_normal.json \
    --seg-model models/seg_4_classes_all_0709_train_aug3/best.pt \
    --device cuda
```

如果想沿用旧 pipeline 的 knob 局部 patch 裁剪，可以额外启用第二个 OBB 模型：

```bash
python msecnet_add_cnn_rgb/infer_pipeline.py \
    msecnet_add_cnn_rgb/ckpt_resnet50_001/best.pt \
    path/to/image.png \
    path/to/cloud.npz \
    output_normal.json \
    --use-obb \
    --obb-model models/inner_obb_clean_v11m_0129/best.pt \
    --device cuda
```

默认不启用 `--use-obb`，因为一个稳定的 inner_cover 分割 mask 已经足够得到内盖法向，也更适合边缘端。

如果不想用 YOLO mask 过滤点云：

```bash
python msecnet_add_cnn_rgb/infer_pipeline.py \
    msecnet_add_cnn_rgb/ckpt_resnet50_001/best.pt \
    path/to/image.png \
    path/to/cloud.npz \
    output_normal.json \
    --no-yolo-mask \
    --device cuda
```

这个模式通常只用于调试；正式推理建议保留 YOLO mask。

### 模型级推理：手动 center

```bash
python msecnet_add_cnn_rgb/infer_e2e.py \
    msecnet_add_cnn_rgb/ckpt_resnet50_001/best.pt \
    path/to/image.png \
    path/to/cloud.npz \
    output_normal.json \
    --device cuda
```

如果已经知道油箱盖 knob center，可以传图像坐标。这个入口不会自动跑 YOLO：

```bash
python msecnet_add_cnn_rgb/infer_e2e.py \
    msecnet_add_cnn_rgb/ckpt_resnet50_001/best.pt \
    path/to/image.png \
    path/to/cloud.npz \
    output_normal.json \
    --knob-center "320,240" \
    --device cuda
```

输出 JSON：

```json
{
  "normal": [0.0, 0.0, 1.0],
  "centroid": [0.0, 0.0, 0.0],
  "points": 1234
}
```

## 和 MoGe 版本对比

| Version | RGB encoder | 推理输入 | 需要特征目录 | 优点 | 缺点 |
|---------|-------------|----------|--------------|------|------|
| `msecnet_add_rgb` | MoGe / DINOv2 ViT-L | 点云 + MoGe特征 | 需要 | 精度强，已有结果 `2.38 deg` | 边缘部署重，链路复杂 |
| `msecnet_add_cnn_rgb` | Light CNN / ResNet | RGB + 点云 | 不需要 | 端到端，部署简单 | 语义先验弱于 MoGe/DINOv2 |

## 建议实验

先跑 ResNet50 默认配置，然后和已有 MoGe-RGB、Light CNN 结果比较：

| Model | RGB encoder | Mean Angular Error | Median | <=10deg |
|-------|-------------|--------------------|--------|---------|
| MSECNet baseline | none | 6.17 deg | - | - |
| MSECNet + MoGe RGB | DINOv2/MoGe | 2.38 deg | 2.25 deg | 100.0% |
| MSECNet + CNN RGB | Light CNN | 3.16 deg | - | - |
| MSECNet + CNN RGB | ResNet50 layer2 | 待填 | 待填 | 待填 |

查看 CNN-RGB 最佳结果：

```bash
python - <<'PY'
import csv
p = "msecnet_add_cnn_rgb/ckpt_resnet50_001/metrics.csv"
with open(p, newline="", encoding="utf-8") as f:
    vals = [r for r in csv.DictReader(f) if r["split"] == "val"]
best = min(vals, key=lambda r: float(r["mean_ang_err"]))
print(best)
PY
```

## 边缘端说明

这个版本仍然依赖：

```text
PyTorch
torchvision（使用 ResNet backbone 时需要）
MSECNet pointops extension
CNN/ResNet 权重
点云 npz 中的 xyz / K_norm / w / h
RGB 图像
```

但不依赖：

```text
MoGe
Hugging Face
DINOv2
data/moge_features
```

如果边缘端没有 CUDA，需要确认 MSECNet 的 pointops 是否支持 CPU。当前项目主要按 CUDA 路径使用。
