# MSECNet + MoGe RGB特征融合

基于 `NORMALNET_RGB_FUSION.md` 中的路径③实现：利用MoGe DINOv2 ViT-L encoder的中间特征进行RGB-几何特征融合。

## 核心思路

1. **特征提取**：从MoGe的DINOv2 encoder提取feature map和CLS token
2. **投影到3D**：将2D特征图投影到3D点云的每个点
3. **全局聚合**：max pool + avg pool + CLS token形成全局RGB特征
4. **门控融合**：与MSECNet的几何特征通过门控机制融合

## 优势

- **避免特征污染**：RGB和几何特征独立聚合，不在邻域内竞争
- **高级语义特征**：DINOv2特征已包含几何语义，不是raw颜色值
- **可预计算**：特征提取可离线完成，训练时无GPU开销
- **互补性强**：MoGe提供相机视角先验，PointNet++提供旋转不变几何

## 使用流程

### 1. 预计算MoGe特征

```bash
# 激活MoGe环境
conda activate moge_env  # 或你的MoGe环境名

# 运行预计算脚本
python msecnet_add_rgb/precompute_moge_feat.py \
    data/pcd_dataset_roi \
    data/moge_features \
    --device cuda
```

输出：`data/moge_features/<sample_name>.npz` 包含：
- `feat_map`: (dim_out, h_low, w_low) encoder特征图
- `cls_token`: (1024,) CLS token
- `h`, `w`: 原图像尺寸

### 2. 训练MSECNet+RGB

```bash
# 激活MSECNet环境
conda activate msecnet_env  # 或你的MSECNet环境名

# 运行训练（soft curriculum）
python msecnet_add_rgb/train_rgb_fusion.py \
    shared/normal_labels_patch03.npz \
    data/pcd_dataset_roi \
    data/moge_features \
    --soft \
    --steps 12000 \
    --bs 12 \
    --max-points 0 \
    --aug-deg 45 \
    --lr 5e-4 \
    --val-every 1000 \
    --out msecnet_add_rgb/ckpt
```

参数说明：
- `--soft`: 使用soft curriculum学习（加权所有样本）
- `--max-points 0`: 使用所有点（不子采样）
- `--aug-deg 45`: 旋转增强角度

### 3. 评估结果

训练过程会输出验证集指标：
- `mean_ang_err`: 平均角度误差（目标：低于6.17°）
- `median`: 中位数误差
- `<=10deg`: 10度以内的比例

## 实现细节

### 特征投影

```python
def project_features_to_points(xyz, feat_map, K_norm, w, h):
    # 3D点 -> 图像坐标 -> 特征图坐标 -> 双线性插值
    uv = project(xyz, K_norm)
    uv_feat = uv * (feat_w/w, feat_h/h)
    return bilinear_sample(feat_map, uv_feat)
```

### 全局聚合

```python
rgb_global = concat([
    max_pool(point_feats),  # (dim_out,)
    avg_pool(point_feats),  # (dim_out,)
    cls_token               # (1024,)
])  # (2*dim_out + 1024,)
```

### 门控融合

```python
alpha = sigmoid(Linear(geom_feat))  # 学习的门控权重
fused = concat(geom_feat, alpha * rgb_feat)
```

## 预期效果

根据 `NORMALNET_RGB_FUSION.md` 分析：

| 方法 | Mean Angular Error | 说明 |
|------|-------------------|------|
| 纯几何 (baseline) | 6.17° | MSECNet v2 |
| Naive RGB fusion | 8.19° | 直接concat，负作用 |
| **MoGe特征融合** | **<6.17° (预期)** | 本方案，理论上应优于baseline |

关键验证点：
1. 是否解决了naive fusion的特征污染问题
2. MoGe语义特征是否提供了有效的外观先验
3. 是否超越了各自的单模态上限

## 文件结构

```
msecnet_add_rgb/
├── precompute_moge_feat.py  # 步骤1: 提取MoGe特征
├── train_rgb_fusion.py      # 步骤2: 训练融合模型
├── ckpt/                     # 模型检查点
│   └── best.pt
└── README.md                 # 本文档
```

## 故障排查

### 问题1: MoGe encoder没有返回features

检查MoGe模型版本，确保使用的是支持encoder输出的版本。可能需要修改`precompute_moge_feat.py`中的特征提取代码。

### 问题2: RGB特征维度不匹配

DINOv2 ViT-L的dim_out通常是768，如果不同需要修改`train_rgb_fusion.py`中的`rgb_feat_dim`：

```python
rgb_feat_dim = dim_out + dim_out + 1024  # max_pool + avg_pool + cls_token
```

### 问题3: 训练显存不足

减小batch size：
```bash
python train_rgb_fusion.py ... --bs 6
```

或启用点云子采样：
```bash
python train_rgb_fusion.py ... --max-points 512
```

## 下一步

1. **对比实验**：
   - 只用CLS token（Step 1）
   - 只用feature map（Step 2）
   - 完整融合（Step 3）

2. **消融实验**：
   - 去掉门控机制
   - 更换池化策略（只max或只avg）
   - 调整RGB特征投影层深度

3. **可视化**：
   - 门控权重α的分布
   - 预测误差 vs RGB特征置信度
