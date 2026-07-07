# NormalNet RGB 融合策略

## 问题分析

### 当前做法的问题

当前 `--rgb` 在 SA1 第一层将 xyz 和 rgb 直接拼接：

```
[xyz(3), rgb(3)] → shared MLP → max pool over neighbors
```

PointNet++ 的 max pooling 语义是"在邻域内为每个特征维度选出最强激活"。xyz 和 rgb 的语义空间正交：
- **xyz** 编码的是局部几何（曲率、朝向、凹凸），max 选出"几何上最显著的特征点"
- **rgb** 编码的是外观属性（颜色、纹理），max 选出"颜色上最突出的点"

两者在同一个 max 下竞争，会互相污染。典型的失败模式：一个深色噪点的颜色值异常激活了某个特征维度，覆盖了几何上更有意义的激活。

### 实验证据

| 配置 | Mean Angular Error | 说明 |
|------|-------------------|------|
| 纯 xyz + soft 课程 | **6.17°** | 最佳 |
| Soft + RGB | 8.19° | RGB 反而干扰 |

**RGB 在 naive fusion 下是负作用**，但这不是 RGB 本身无用，而是融合方式有问题。

---

## 可行的 RGB 融合路径

### 路径 ①：Per-point MLP → Global Pooling

```
每个点的 RGB(3,) → 共享MLP(3→32→64) → max_pool over all 1024 points → rgb_global(64,)
                                                                              │
                                    ┌─ PointNet++(xyz) → geom_feat(1024,) ────┼─→ concat → MLP head
                                    │                                          │
```

**原理**：继承原始 PointNet 处理 per-point feature 的方式——不做层级聚合，纯 per-point 编码 + 全局池化。RGB 特征独立聚合后再与几何特征融合。

**优点**：
- 改动极小（~20 行）
- rgb 和 xyz 在独立的 max pool 中聚合，互不干扰
- 全局纹理统计量（平均色调、对比度）可能对 cap 就够用了

**缺点**：表达能力弱。cap 表面颜色均匀时，per-point MLP 可能只学到"整体亮度"一个自由度。

---

### 路径 ②：2D CNN on RGB → Project to 3D Points

```
RGB图像 (112, 112, 3) → 轻量CNN(3→32→64→128) → feature map (112, 112, 128)
                                                          │
                       按每点的 (u, v) 像素坐标索引 ───────┘
                                                          │
                       每点取对应位置的 128D 特征 → max_pool → rgb_global(128,)
```

**原理**：RGB 的天然表示是 2D 格点，CNN 的平移不变性和局部感受野对纹理是最优的。比 ball query 在 3D 里找颜色邻居合理得多——两个 3D 相邻的点可能在像素上差很远（比如边缘两侧），但 CNN 在 2D 上可以捕获纹理梯度。

**优点**：
- 利用 CNN 对纹理的归纳偏置
- 项目已有 `d["K_norm"], d["w"], d["h"]`，投影零成本
- 和 PointNet++ 完全解耦，各自处理最适合的表示

**缺点**：
- 需要 batch 内图像尺寸一致，或做 resize/pad
- 轻量 CNN 表达能力有限，但过重则过拟合纹理

---

### 路径 ③：MoGe Encoder 中间特征 → Project + Pool（**推荐**）

```
RGB图像 (H, W, 3)
       │
       ├──→ MoGe encoder (DINOv2 ViT-L, frozen)
       │         │
       │         ├──→ encoder features: (B, dim_out, base_h, base_w)
       │         │         │
       │         │         ├──→ bilinear resize → (B, dim_out, H, W)
       │         │         │
       │         │         ├──→ 按 (u,v) 索引每点 → (B, N, dim_out)
       │         │         │
       │         │         └──→ max_pool + avg_pool → appr_global(2*dim_out,)
       │         │                                                │
       │         └──→ CLS token: (B, 1024) ──────────────────────→ concat → MLP head
       │                                                                      │
       └──→ 反投影 → xyz (B, 3, N) → PointNet++ → geom_global(1024,) ────────┘
```

#### MoGe 中间层为什么适合

MoGe 的 encoder 是 DINOv2 ViT-L（24 层 transformer，embed_dim=1024），在大规模数据上预训练。其 `forward()` 内部结构（文件：`moge/model/v2.py`）：

1. **Encoder**（`self.encoder(image)`）：
   - Patch embed：`(B, 3, H, W) → (B, N, 1024)`
   - 24 层 transformer blocks
   - **取最后 4 层输出**，每层 `(B, N, 1024)` → 1x1 Conv 投影 → 求和 → `(B, dim_out, base_h, base_w)`
   - 返回 `(features_map, cls_token)`

2. **CLS token**：`(B, 1024)` — 全局图像语义（光照、场景类型、相机-物体关系）

3. **Neck + Heads**：features_map 经 FPN neck → points_head/normal_head/mask_head 输出最终结果

**关键洞察**：MoGe encoder 的中间特征 `(B, dim_out, base_h, base_w)` 已经编码了"每个像素的几何语义"——它在下游被解码为 metric 点云和表面法向。这些特征比 raw RGB 更接近任务目标（法向回归），但比 MoGe 最终输出的 per-pixel normal 保留了更多中间信息。

#### 不同 hook 点的选择

| Hook 点 | 位置 | 形状 | 语义 | 开销 |
|----------|------|------|------|------|
| Encoder 特征图 | `self.encoder()` 返回值 | `(B, dim_out, h_low, w_low)` | 融合了 4 层 transformer 的语义特征 | 无额外计算 |
| Neck 输出 (level 0) | `self.neck(features)[0]` | `(B, dim_neck, h_low, w_low)` | 经 FPN 增强的 coarse feature | + neck forward |
| 任一 head 的中间层 | head ConvStack 中间 | varies | 任务专用特征（如 normal_head 的中间层） | + neck + partial head |

**推荐 hook 点**：Encoder 特征图。理由：
1. 零额外计算（encoder 本来就是 `infer()` 的第一步）
2. 特征已有几何语义（MoGe 用它们解码 normal/points），但不被单一任务过度特化
3. 低分辨率 `(base_h, base_w)` 约 112×112，投影开销小

#### 投影到 3D 点

```
for b in batch:
    # MoGe infer 第一步（不跑 neck/heads）
    feat_map, cls_token = model.encoder(image[b])          # (dim_out, h_low, w_low), (1024,)
    feat_map = interpolate(feat_map, (H, W))                # resize 到原图分辨率

    for each point j in cloud_b:
        u_j, v_j = point_to_pixel(xyz_j, K_norm, w, h)     # 点在图像上的投影坐标
        feat_j = bilinear_sample(feat_map, u_j, v_j)        # (dim_out,)
    
    appr_feat = concat(max_pool(all feat_j), avg_pool(all feat_j))  # (2*dim_out,)
    appr_feat += cls_token                                          # (2*dim_out+1024,)
```

#### 与 PointNet++ 几何特征的融合

```
geom_feat (1024,) ─┐
                    ├─→ concat → FC(1024+2*dim_out+1024 → 512) → ... → (3,)
appr_feat (...)  ──┘
```

或加一个门控机制，让网络自适应地决定用多少外观信息：

```
alpha = sigmoid(FC(geom_feat))           # learnable gate
fused = concat(geom_feat, alpha * appr_feat) → MLP head
```

#### 为什么预期比 raw RGB fusion 好

| 维度 | raw RGB (当前) | MoGe 中间特征 |
|------|---------------|---------------|
| 特征语义 | 原始颜色值 | 几何相关的语义（纹理→形状的中间表示） |
| 融合方式 | 点级 concat + max（冲突） | 全局 concat（各自 pool 后融合） |
| 光照不变性 | 依赖颜色抖动增强 | DINOv2 预训练已有部分光照不变性 |
| 泛化能力 | 弱（可能过拟合训练集的颜色分布） | 强（DINOv2 在大规模数据上预训练） |
| 计算开销 | 无 | encoder forward（推理本来就跑）+ 投影 |

#### 实现注意事项

1. **MoGe encoder 是 frozen 的**：不参与 NormalNet 训练的反向传播，只做特征提取。这避免了破坏 DINOv2 预训练权重，也使得训练时只需做一次 encoder forward + 保存特征到 npz。

2. **预计算策略**：训练前可以离线跑 MoGe encoder 把所有样本的 `(feat_map, cls_token)` 存到 npz，训练时只做索引+投影+pool，完全不需要 on-the-fly MoGe forward。

3. **特征维度控制**：`dim_out` 由 MoGe checkpoint 决定（v2 vitl 约 768）。如果太大，可以在 concat 前加一层 FC 降维。

4. **旋转增强下的特征行为**：PointNet++ 分支的 xyz 被旋转，但 MoGe 特征来自 RGB→encoder，**不随点云旋转**。这恰好是我们想要的——MoGe 提供的是"相机绝对朝向"的弱先验（这个视角下 cap 大概朝向哪），而 PointNet++ 提供"形状在任意方向下的法向"。两者互补而非冗余。

---

### 路径 ④：跨模态交叉注意力

```
geom_feat (B, 1024,) → FC_Q → Q (B, 1, d_k)
appr_feat (B, N, dim_out) → FC_K, FC_V → K, V (B, N, d_k)
attention = softmax(Q @ K^T / sqrt(d_k))   # 几何 query 在 N 个点的外观特征上做 attention
fused = attention @ V
```

让几何分支主动"寻找"相关的颜色 cues。对 cap 这个任务来说可能 overkill——纹理弱，大部分点的 RGB 没信息量。

---

## 实施建议

### 推荐路径：③（MoGe Encoder 中间特征）

理由：
1. **项目已有 MoGe**，不需要新依赖
2. **改动量可控**（~80 行：encoder hook + 投影 + new dataset 字段 + 模型修改）
3. **有明确的理论优势**——DINOv2 特征的语义层级远高于 raw RGB
4. **可预计算**，训练不增加 GPU 负担

### 验证方案

分步验证，每一步都可量化：

```
Step 0: 基线 → 纯 xyz, soft 课程 (6.17°)

Step 1: 加 MoGe CLS token only (1024D)
        → 验证全局上下文是否有帮助

Step 2: 加 MoGe encoder feature map (project + pool)
        → 验证 per-point 语义特征是否有帮助

Step 3: 对比 raw RGB fusion (当前 --rgb = 8.19°)
        → 验证 MoGe 特征是否解决了 naive RGB fusion 的问题

Step 4: 对比 MoGe-only baseline (precompute_moge_normal.py = ~median agree deg)
        → 验证 PointNet++ + MoGe 的融合是否超越了各自的单模态上限
```

### 主要改动清单

| 文件 | 改动 |
|------|------|
| `pose/normal_net.py` | `NormalNet` 增加 `appr_feat_dim` 参数和融合头；`CapNormalDS` 增加读取预计算 MoGe 特征的逻辑 |
| `pose/precompute_moge_feat.py`（新增） | 离线运行 MoGe encoder，把每帧的 `(feat_map, cls_token)` 存入 npz 或独立目录 |
| `pose/NORMALNET.md` | 更新实验结果 |
