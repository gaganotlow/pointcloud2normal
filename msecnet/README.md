# MSECNet 燃油盖内盖法向量估计

本目录只保留当前 MSECNet 的数据处理、训练、测试集推理和网页查看流程。所有命令均使用 `point2normal` Conda 环境。

```bash
cd /data2/shendu/code/ruoyu/train_point2normal
```

## 当前产物

| 项目 | 路径 | 说明 |
|---|---|---|
| 已处理数据集 | `data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/` | 局部 pseudo-OBB 点云面片与人工法向量标签 |
| 历史基线权重 | `msecnet/out/ckpt_msecnet_v4_manual_pseudo_obb/best.pt` | 旧版无向目标训练的 checkpoint；保留作对照 |
| 历史基线报告 | `msecnet/out/ckpt_msecnet_v4_manual_pseudo_obb/inference_test/report.json` | 旧版逐样本预测与汇总指标 |

模型只预测法向量方向。网页中的点云起点、蓝色矩形大小和面内切线均来自
`anchors_manual3d.json` 的人工锚点，不是模型输出。

## 数据集处理

源数据集提供人工复核后的 3D 矩形和法向量。`prepare_pseudo_obb_dataset.py` 会校验标注，按 pseudo-OBB 截取局部棱柱点云，限制每个面片的最大点数，并按车型划分互不重叠的训练、验证、测试集。

```bash
conda run --no-capture-output -n point2normal python \
  msecnet/prepare_pseudo_obb_dataset.py \
  data/fuelcap_pass_20260717_5873 \
  data/manual_pseudo_obb_new \
  --max-points 4096 \
  --obb-expand 2.0 \
  --obb-half-depth-m 0.005
```

生成目录包含：

| 文件 | 说明 |
|---|---|
| `clouds/*.npz` | 训练使用的局部点云面片 |
| `labels_manual3d.npz` | 样本文件名和人工目标法向量 |
| `anchors_manual3d.json` | 人工中心、矩形尺寸、切线和法向量 |
| `split_by_generalization_group.json` | 按真实车型或采集会话隔离、样本量平衡的训练/验证/测试列表 |
| `manifest.jsonl` | 每个样本的源数据追溯信息与划分 |

导出一个已处理面片，与源点云进行对照：

```bash
conda run --no-capture-output -n point2normal python \
  msecnet/export_pseudo_obb_ply.py \
  data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb \
  --file '_未分类_易车网__易车网_data0_2_2019款PlusA8L50TFSIquattro舒适型_25.npz' \
  --raw-points 50000
```

## 训练

MSECNet 为面片中的每个点预测向量，但部署目标是每个面片的一条**有向**法向量。人工数据已校验法向量朝向相机（`normal dot (-center) > 0`），因此训练和评估均保留方向：先归一化每点输出，再平均并归一化为面片预测。

训练目标以面片级余弦损失为主，直接优化部署输出；默认以 0.25 权重加入逐点余弦损失，约束点预测的局部一致性。报告中的 `point_consensus` 是归一化点向量平均后的长度，范围为 0 到 1，可用于识别点预测相互抵消的低置信度样本。

`split_by_generalization_group.json` 是用于下一次从头训练的新划分。它与历史 `split_by_car_model.json` 不同，不能拿来评估已经按历史划分训练的 checkpoint，否则新的验证/测试集会包含旧训练样本。

```bash
conda run --no-capture-output -n point2normal python msecnet/train.py \
  data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/labels_manual3d.npz \
  data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/clouds \
  --centers data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/anchors_manual3d.json \
  --split data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/split_by_generalization_group.json \
  --steps 70000 \
  --bs 24 \
  --max-points 1024 \
  --radius 0.3 \
  --aug-deg 45 \
  --point-loss-weight 0.25 \
  --seed 20260722 \
  --early-stop-patience 100 \
  --snapshot-every 1000 \
  --out msecnet/out/manual_pseudo_obb_oriented
```

当前数据集共有 5,864 个样本：训练集 4,688，验证集 589，测试集 587。

## 测试集推理

不带参数运行时，`infer.py` 会使用当前最优权重在保留的测试集上评估，并在 checkpoint 目录生成 `report.json` 和 `predictions.csv`。

```bash
conda run --no-capture-output -n point2normal python msecnet/infer.py
```

历史无向基线测试结果（新训练应以有向角误差重新报告，不应与此数值直接混用）：

| 样本数 | 平均轴向误差 | 中位轴向误差 | 轴向误差 <= 10 deg |
|---:|---:|---:|---:|
| 587 | 2.407 deg | 2.000 deg | 99.1% |

## 网页查看

评估模式下网页为只读。页面会在源点云上同时显示人工目标法向量与模型预测法向量。蓝色矩形来自人工锚点，用于观察平面贴合效果，不属于模型预测。

```bash
conda run --no-capture-output -n point2normal python web_label/server.py \
  --msecnet-report msecnet/out/ckpt_msecnet_v4_manual_pseudo_obb/inference_test/report.json \
  --msecnet-dataset data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb \
  --port 8766
```

浏览器打开 `http://localhost:8766`。追加 `--focus-file FILE.npz` 可只查看一个样本。
