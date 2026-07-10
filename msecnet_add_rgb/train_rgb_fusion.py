#!/usr/bin/env python3
"""MSECNet + MoGe RGB特征融合训练脚本。
将MoGe DINOv2 encoder特征投影到点云并融合到MSECNet进行法向量回归。

Usage: python train_rgb_fusion.py <labels> <pcd_dir> <moge_feat_dir> [options]
"""
import argparse
import copy
import csv
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MROOT = os.path.join(HERE, "..", "msecnet", "MSECNet")
sys.path.insert(0, os.path.join(MROOT, "model"))
sys.path.insert(0, os.path.join(MROOT, "scripts"))
POINTOPS_ROOT = os.path.join(MROOT, "scripts", "lib", "pointops")
for path in glob.glob(os.path.join(POINTOPS_ROOT, "build", "lib.*")):
    sys.path.insert(0, path)
sys.path.insert(0, os.path.join(ROOT, "shared"))
from util import config
from architectures import MSECNet
import cap_patch


RGB_MODES = ("full", "map", "cls")


def rand_rot(rs, max_deg=180.0):
    ax = rs.normal(size=3)
    ax /= np.linalg.norm(ax) + 1e-9
    a = np.radians(rs.uniform(0, max_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)).astype(np.float32)


def project_features_to_points(xyz, feat_map, K_norm, w, h):
    """将2D特征图投影到3D点云。

    Args:
        xyz: (N, 3) 点云坐标
        feat_map: (C, H, W) 特征图
        K_norm: (3, 3) 归一化相机内参
        w, h: 原图像宽高

    Returns:
        point_feats: (N, C) 每个点对应的特征
    """
    feat_map = np.asarray(feat_map, dtype=np.float32)
    C, feat_h, feat_w = feat_map.shape
    N = len(xyz)
    point_feats = np.zeros((N, C), dtype=np.float32)
    if N == 0:
        return point_feats

    xyz = np.asarray(xyz, dtype=np.float32)
    K_norm = np.asarray(K_norm, dtype=np.float32)
    z = xyz[:, 2]

    # K_norm is normalized by image size in the cached MoGe outputs.
    u = K_norm[0, 0] * float(w) * xyz[:, 0] / (z + 1e-9) + K_norm[0, 2] * float(w)
    v = K_norm[1, 1] * float(h) * xyz[:, 1] / (z + 1e-9) + K_norm[1, 2] * float(h)
    valid = (z > 1e-6) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    if not np.any(valid):
        return point_feats

    scale_x = (feat_w - 1) / max(float(w - 1), 1.0)
    scale_y = (feat_h - 1) / max(float(h - 1), 1.0)
    uf = np.clip(u[valid] * scale_x, 0, feat_w - 1)
    vf = np.clip(v[valid] * scale_y, 0, feat_h - 1)

    x0 = np.floor(uf).astype(np.int32)
    y0 = np.floor(vf).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, feat_w - 1)
    y1 = np.clip(y0 + 1, 0, feat_h - 1)
    wx = (uf - x0).astype(np.float32)
    wy = (vf - y0).astype(np.float32)

    f00 = feat_map[:, y0, x0].T
    f01 = feat_map[:, y1, x0].T
    f10 = feat_map[:, y0, x1].T
    f11 = feat_map[:, y1, x1].T
    sampled = ((1 - wx)[:, None] * (1 - wy)[:, None] * f00 +
               (1 - wx)[:, None] * wy[:, None] * f01 +
               wx[:, None] * (1 - wy)[:, None] * f10 +
               wx[:, None] * wy[:, None] * f11)

    point_feats[valid] = sampled.astype(np.float32)
    return np.nan_to_num(point_feats, copy=False)


def rgb_dim_from_parts(feat_dim, cls_dim, mode):
    if mode == "full":
        return 2 * int(feat_dim) + int(cls_dim)
    if mode == "map":
        return 2 * int(feat_dim)
    if mode == "cls":
        return int(cls_dim)
    raise ValueError(f"unknown rgb mode: {mode}")


def infer_moge_feature_dims(moge_feat_dir, files=None):
    """Return (feature-map channels, cls-token dim) from saved MoGe feature files."""
    candidates = []
    if files is not None:
        candidates.extend(os.path.join(moge_feat_dir, str(f)) for f in files)
    if os.path.isdir(moge_feat_dir):
        candidates.extend(os.path.join(moge_feat_dir, f)
                          for f in sorted(os.listdir(moge_feat_dir))
                          if f.endswith(".npz") and not f.endswith(".tmp.npz"))

    seen = set()
    for path in candidates:
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            with np.load(path) as d:
                feat_dim = int(d["feat_map"].shape[0]) if "feat_map" in d else 0
                cls_dim = int(d["cls_token"].shape[0]) if "cls_token" in d else 0
        except Exception:
            continue
        if feat_dim > 0 or cls_dim > 0:
            return feat_dim, cls_dim
    raise FileNotFoundError(
        f"no usable MoGe feature npz found in {moge_feat_dir}; run precompute_moge_feat.py first"
    )


def build_rgb_global(xyz, cloud_npz, moge_feat_path, rgb_feat_dim, mode="full", allow_missing=False):
    """Build one fixed-length RGB/MoGe descriptor for a point patch."""
    if mode not in RGB_MODES:
        raise ValueError(f"mode must be one of {RGB_MODES}, got {mode}")
    if len(xyz) == 0:
        return np.zeros(rgb_feat_dim, dtype=np.float32)
    if not os.path.exists(moge_feat_path):
        if not allow_missing:
            raise FileNotFoundError(f"missing MoGe feature file: {moge_feat_path}")
        return np.zeros(rgb_feat_dim, dtype=np.float32)

    with np.load(moge_feat_path) as moge_data:
        parts = []
        if mode in ("full", "map"):
            feat_map = moge_data["feat_map"].astype(np.float32)
            point_feats = project_features_to_points(
                xyz, feat_map, cloud_npz["K_norm"], int(cloud_npz["w"]), int(cloud_npz["h"])
            )
            parts.extend([point_feats.max(axis=0), point_feats.mean(axis=0)])
        if mode in ("full", "cls"):
            parts.append(moge_data["cls_token"].astype(np.float32))

    rgb_global = np.concatenate(parts, axis=0).astype(np.float32)
    rgb_global = np.nan_to_num(rgb_global, copy=False)
    if rgb_global.shape[0] != rgb_feat_dim:
        raise ValueError(
            f"MoGe feature dim mismatch for {moge_feat_path}: got {rgb_global.shape[0]}, "
            f"expected {rgb_feat_dim}"
        )
    return rgb_global


def offset_to_counts(offset):
    start = torch.cat([offset.new_zeros(1), offset[:-1]])
    return (offset - start).long()


def missing_moge_feature_files(moge_feat_dir, files):
    return [str(f) for f in files if not os.path.exists(os.path.join(moge_feat_dir, str(f)))]


def log_print(out_dir, msg):
    print(msg, flush=True)
    with open(os.path.join(out_dir, "train.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


METRIC_FIELDS = [
    "time", "step", "split", "loss", "ema_loss", "lr", "mean_ang_err",
    "median_ang_err", "p10", "steps_per_sec", "best_mean_ang_err"
]


def append_metrics_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in METRIC_FIELDS})


def record_metric(out_dir, history, row):
    row = dict(row)
    row.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    history.append(row)
    append_metrics_csv(os.path.join(out_dir, "metrics.csv"), row)
    append_jsonl(os.path.join(out_dir, "metrics.jsonl"), row)


def safe_stem(name, max_len=90):
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    stem = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", stem, flags=re.UNICODE)
    return stem[:max_len] or "sample"


def try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable, skip plots: {exc}", flush=True)
        return None


def plot_training_curves(history, out_dir):
    plt = try_import_matplotlib()
    if plt is None:
        return
    train = [r for r in history if r.get("split") == "train"]
    val = [r for r in history if r.get("split") == "val"]
    if not train and not val:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=False)
    if train:
        xs = [int(r["step"]) for r in train]
        axes[0].plot(xs, [float(r["ema_loss"]) for r in train], label="ema_loss", color="#1f77b4")
        axes[0].plot(xs, [float(r["loss"]) for r in train], label="loss", color="#9ecae1", alpha=0.45)
        axes[0].set_ylabel("train loss")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
    if val:
        xs = [int(r["step"]) for r in val]
        axes[1].plot(xs, [float(r["mean_ang_err"]) for r in val], marker="o", label="mean", color="#d62728")
        axes[1].plot(xs, [float(r["median_ang_err"]) for r in val], marker="o", label="median", color="#ff7f0e")
        axes[1].set_ylabel("angle error (deg)")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        axes[2].plot(xs, [float(r["p10"]) for r in val], marker="o", label="<=10deg %", color="#2ca02c")
        axes[2].set_ylabel("validation %")
        axes[2].set_xlabel("step")
        axes[2].legend()
        axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=160)
    plt.close(fig)


def load_patch_xyz(npz_file, pcd_dir, kc_dict, radius, max_points=2500):
    with np.load(os.path.join(pcd_dir, npz_file)) as d:
        xyz = d["xyz"][d["label"] == 1].astype(np.float32)
        kc = kc_dict.get(npz_file)
        if kc is not None and len(xyz) >= 120:
            _, pm = cap_patch.extract(xyz, d["K_norm"], int(d["w"]), int(d["h"]), kc, radius_frac=radius)
            if pm.sum() >= 80:
                xyz = xyz[pm]
        if len(xyz) == 0:
            xyz = d["xyz"].astype(np.float32)
    xyz = xyz - xyz.mean(0)
    xyz = xyz / (np.linalg.norm(xyz, axis=1).max() + 1e-9)
    if max_points > 0 and len(xyz) > max_points:
        rs = np.random.default_rng(0)
        xyz = xyz[rs.choice(len(xyz), max_points, replace=False)]
    return xyz


def plot_validation_error_summary(details, step, out_dir):
    plt = try_import_matplotlib()
    if plt is None or not details:
        return
    vis_dir = os.path.join(out_dir, "val_vis")
    os.makedirs(vis_dir, exist_ok=True)
    errs = np.array([d["error_deg"] for d in details], dtype=np.float32)
    order = np.argsort(errs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(errs, bins=30, color="#4c78a8", alpha=0.85)
    axes[0].axvline(float(errs.mean()), color="#e45756", label=f"mean {errs.mean():.2f}")
    axes[0].axvline(float(np.median(errs)), color="#f58518", label=f"median {np.median(errs):.2f}")
    axes[0].set_xlabel("angle error (deg)")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(np.arange(len(errs)), errs[order], color="#54a24b")
    axes[1].set_xlabel("validation samples sorted by error")
    axes[1].set_ylabel("angle error (deg)")
    axes[1].grid(alpha=0.2)

    fig.suptitle(f"Validation error at step {step}")
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, f"step_{step:06d}_error_summary.png"), dpi=160)
    plt.close(fig)


def plot_sample_cloud_normal(detail, xyz, out_path, title=None):
    plt = try_import_matplotlib()
    if plt is None:
        return
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pred = np.asarray(detail["pred"], dtype=np.float32)
    target = np.asarray(detail["target"], dtype=np.float32)
    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=2, alpha=0.35, c=xyz[:, 2], cmap="viridis")
    ax.quiver(0, 0, 0, target[0], target[1], target[2], length=0.7, color="#2ca02c", linewidth=2, label="target")
    ax.quiver(0, 0, 0, pred[0], pred[1], pred[2], length=0.7, color="#d62728", linewidth=2, label="pred")
    ax.set_title(title or f"validation sample\nerr={detail['error_deg']:.2f}deg")
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feature_heatmap(npz_file, moge_feat_dir, out_path, title=None):
    plt = try_import_matplotlib()
    if plt is None:
        return
    path = os.path.join(moge_feat_dir, npz_file)
    if not os.path.exists(path):
        return
    with np.load(path) as d:
        if "feat_map" not in d:
            return
        fmap = d["feat_map"].astype(np.float32)
    heat = np.linalg.norm(fmap, axis=0)
    lo, hi = np.percentile(heat, [2, 98])
    heat = np.clip((heat - lo) / (hi - lo + 1e-9), 0, 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(heat, cmap="magma", interpolation="nearest")
    ax.set_title(title or "MoGe feature-map norm")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_validation_visuals(details, step, out_dir, pcd_dir, moge_feat_dir, kc_dict, radius, num_samples):
    if not details or num_samples <= 0:
        return
    step_dir = os.path.join(out_dir, "val_vis", f"step_{step:06d}")
    os.makedirs(step_dir, exist_ok=True)
    ranked = sorted(details, key=lambda d: d["error_deg"], reverse=True)
    selected = ranked[:num_samples]
    for rank, detail in enumerate(selected, 1):
        name = detail["file"]
        stem = f"{rank:02d}_err_{detail['error_deg']:.2f}_{safe_stem(name)}"
        try:
            xyz = load_patch_xyz(name, pcd_dir, kc_dict, radius)
            title = f"rank {rank} err={detail['error_deg']:.2f}deg"
            plot_sample_cloud_normal(detail, xyz, os.path.join(step_dir, stem + "_normal.png"), title=title)
            plot_feature_heatmap(name, moge_feat_dir, os.path.join(step_dir, stem + "_featmap.png"), title=title)
        except Exception as exc:
            print(f"WARNING: failed to visualize {name}: {exc}", flush=True)


class DSWithRGB(Dataset):
    """带MoGe RGB特征的数据集。"""

    def __init__(self, files, normals, pcd_dir, moge_feat_dir, kc, radius,
                 train, max_points, rgb_feat_dim, weights=None, aug_deg=0.0,
                 rgb_mode="full", allow_missing_rgb=False, return_name=False):
        self.files = files
        self.normals = normals
        self.pcd_dir = pcd_dir
        self.moge_feat_dir = moge_feat_dir
        self.kc = kc
        self.radius = radius
        self.train = train
        self.maxp = max_points
        self.rgb_feat_dim = rgb_feat_dim
        self.weights = weights
        self.aug_deg = aug_deg
        self.rgb_mode = rgb_mode
        self.allow_missing_rgb = allow_missing_rgb
        self.return_name = return_name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        # 加载点云数据
        d = np.load(os.path.join(self.pcd_dir, self.files[i]))
        xyz = d["xyz"][d["label"] == 1].astype(np.float32)
        n = self.normals[i].astype(np.float32).copy()

        # 提取knob patch
        kc = self.kc.get(self.files[i])
        if kc is not None and len(xyz) >= 120:
            _, pm = cap_patch.extract(xyz, d["K_norm"], int(d["w"]), int(d["h"]),
                                     kc, radius_frac=self.radius)
            if pm.sum() >= 80:
                xyz = xyz[pm]

        if len(xyz) == 0:
            xyz = d["xyz"].astype(np.float32)

        # 随机子采样
        rs = np.random.RandomState((i * 7919 + int(time.time() * 1000)) & 0x7fffffff if self.train else i)
        if self.maxp > 0 and len(xyz) > self.maxp:
            idx = rs.choice(len(xyz), self.maxp, replace=False)
            xyz = xyz[idx]

        moge_feat_path = os.path.join(self.moge_feat_dir, str(self.files[i]))
        rgb_global = build_rgb_global(
            xyz, d, moge_feat_path, self.rgb_feat_dim, self.rgb_mode, self.allow_missing_rgb
        )

        # 中心化和归一化
        xyz = xyz - xyz.mean(0)
        xyz = xyz / (np.linalg.norm(xyz, axis=1).max() + 1e-9)

        # 训练时增强
        if self.train:
            R = rand_rot(rs, self.aug_deg)
            xyz = xyz @ R.T
            n = R @ n
            xyz = xyz + rs.normal(0, 0.01, xyz.shape).astype(np.float32)

        n = n / (np.linalg.norm(n) + 1e-9)
        w = float(self.weights[i]) if self.weights is not None else 1.0

        sample = (xyz.astype(np.float32), n.astype(np.float32), np.float32(w), rgb_global)
        if self.return_name:
            return sample + (str(self.files[i]),)
        return sample


def collate(batch):
    coords = [torch.from_numpy(b[0]) for b in batch]
    counts = torch.tensor([len(c) for c in coords], dtype=torch.int64)
    coord = torch.cat(coords, 0).float()
    offset = torch.cumsum(counts, 0).int()
    normal = torch.from_numpy(np.stack([b[1] for b in batch])).float()
    w = torch.tensor([b[2] for b in batch]).float()
    rgb_global = torch.from_numpy(np.stack([b[3] for b in batch])).float()
    if len(batch[0]) > 4:
        names = [b[4] for b in batch]
        return coord, offset, normal, w, counts, rgb_global, names
    return coord, offset, normal, w, counts, rgb_global


class MSECNetWithRGB(nn.Module):
    """MSECNet + RGB特征融合头。"""

    def __init__(self, cfg, rgb_feat_dim):
        super().__init__()
        self.msecnet = MSECNet(cfg)

        # RGB特征融合头
        geom_dim = self.msecnet.classifier[0].in_features
        self.geom_dim = geom_dim
        self.rgb_feat_dim = int(rgb_feat_dim)
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_feat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(geom_dim, 1),
            nn.Sigmoid()
        )

        # 融合后分类器
        self.fusion_classifier = nn.Sequential(
            nn.Linear(geom_dim + 512, geom_dim),
            nn.BatchNorm1d(geom_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(geom_dim, cfg.num_classes)
        )

    def forward(self, p, x, o, rgb_global):
        # MSECNet几何特征提取（不包括最后的分类层）
        # 需要获取倒数第二层的特征
        p_from_encoder = []
        x_from_encoder = []
        o_from_encoder = []
        side_output = []

        # encoder
        for block_i, block in enumerate(self.msecnet.encoder_blocks):
            if block_i in self.msecnet.encoder_skips:
                p_from_encoder.append(p)
                x_from_encoder.append(x)
                o_from_encoder.append(o)
                side_output.append([p, x, o])
            p, x, o = block(p, x, o)
        side_output.append([p, x, o])

        # decoder
        for block_i, block in enumerate(self.msecnet.decoder_blocks):
            if block_i in self.msecnet.decoder_upsample:
                p_dense = p_from_encoder.pop()
                x_dense = x_from_encoder.pop()
                o_dense = o_from_encoder.pop()
                p, x, o = block(p_dense, x_dense, o_dense, p, x, o)
            else:
                p, x, o = block(p, x, o)

        # MSEC branch
        p_dense, x_dense, o_dense = side_output[0]
        ms_feat = x_dense
        for i in range(1, len(side_output[:self.msecnet.n_scale])):
            p_sp, x_sp, o_sp = side_output[i]
            from lib.pointops.functions import pointops
            interpolated = pointops.interpolation_flexible(
                p_sp, p_dense, x_sp, o_sp, o_dense,
                k=self.msecnet.nsample_interp,
                weight_type=self.msecnet.interp_weight_type
            )
            ms_feat = torch.cat([ms_feat, interpolated], dim=1)

        ms_feat_new = self.msecnet.ms_fusion(p_dense, ms_feat, o_dense)[1]
        ms_edge = self.msecnet.edge_transfrom(p_dense, ms_feat_new, o_dense)[1]
        geom_feat = self.msecnet.ee(x, ms_edge)  # (N, geom_dim)
        if geom_feat.shape[1] != self.geom_dim:
            raise RuntimeError(f"MSECNet feature dim changed: got {geom_feat.shape[1]}, expected {self.geom_dim}")

        # RGB特征投影
        rgb_feat = self.rgb_proj(rgb_global)  # (B, 512)

        # 按offset展开RGB特征到每个样本的点，支持batch内变长点云。
        counts = offset_to_counts(o)
        if rgb_feat.shape[0] != counts.shape[0]:
            raise RuntimeError(f"rgb batch={rgb_feat.shape[0]} but offsets={counts.shape[0]}")
        if int(counts.sum().item()) != geom_feat.shape[0]:
            raise RuntimeError(f"point count mismatch: offsets sum={int(counts.sum().item())}, geom={geom_feat.shape[0]}")
        rgb_feat_expanded = rgb_feat.repeat_interleave(counts.to(rgb_feat.device), dim=0)  # (N, 512)

        # 门控融合
        alpha = self.gate(geom_feat)  # (N, 1)
        fused_feat = torch.cat([geom_feat, alpha * rgb_feat_expanded], dim=1)  # (N, geom_dim+512)

        # 最终分类
        out = self.fusion_classifier(fused_feat)
        return out


class ModelEMA:
    """Exponential moving average of model weights for stabler validation/inference."""

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # Ramp up EMA early so the averaged weights do not lag too far behind.
        decay = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        model_state = model.state_dict()
        for key, ema_value in self.ema.state_dict().items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


def set_optimizer_lr(opt, lr):
    for group in opt.param_groups:
        group["lr"] = float(lr)


def get_optimizer_lr(opt):
    return float(opt.param_groups[0]["lr"])


def cosine_lr(step, total_steps, base_lr, min_lr, warmup_steps, warmup_start_factor):
    """1-based step cosine schedule with linear warmup and non-zero LR floor."""
    step = int(step)
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    if warmup_steps > 0 and step <= warmup_steps:
        alpha = step / float(warmup_steps)
        return base_lr * (warmup_start_factor + alpha * (1.0 - warmup_start_factor))

    span = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / float(span), 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    errs = []
    details = []
    for batch in loader:
        if len(batch) == 7:
            coord, offset, normal, w, counts, rgb_global, names = batch
        else:
            coord, offset, normal, w, counts, rgb_global = batch
            names = [None] * len(counts)
        coord = coord.to(dev)
        offset = offset.to(dev)
        rgb_global = rgb_global.to(dev)
        normal = normal.to(dev)

        pp = F.normalize(model(coord, torch.zeros(coord.shape[0], 0, device=dev),
                              offset, rgb_global), dim=1)
        idx = 0
        for b, c in enumerate(counts.tolist()):
            agg = F.normalize(pp[idx:idx + c].mean(0), dim=0)
            idx += c
            target = F.normalize(normal[b], dim=0)
            dot = float(torch.dot(agg, target).clamp(-1, 1))
            cos = abs(dot)
            err = float(np.degrees(np.arccos(np.clip(cos, 0, 1))))
            errs.append(err)
            pred = agg.detach().cpu().numpy().astype(float)
            target_np = target.detach().cpu().numpy().astype(float)
            if dot < 0:
                pred = -pred
            details.append({
                "file": names[b],
                "error_deg": err,
                "pred": pred.tolist(),
                "target": target_np.tolist(),
                "points": int(c),
            })

    e = np.array(errs)
    return e.mean(), np.median(e), (e <= 10).mean() * 100, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("pcd_dir")
    ap.add_argument("moge_feat_dir", help="MoGe特征目录")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--max-points", type=int, default=0)
    ap.add_argument("--inlier", type=float, default=0.8)
    ap.add_argument("--agree", type=float, default=15)
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--soft", action="store_true")
    ap.add_argument("--aug-deg", type=float, default=0.0,
                    help="点云/法向旋转增强角度；RGB融合默认0，避免破坏相机视角特征对齐")
    ap.add_argument("--rgb-mode", choices=RGB_MODES, default="full",
                    help="full=max/avg投影特征+CLS, map=只用投影特征, cls=只用CLS token")
    ap.add_argument("--allow-missing-rgb", action="store_true",
                    help="允许缺失MoGe特征文件并以零向量兜底；默认报错以保证实验可解释")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--sched", choices=("onecycle", "cosine", "constant"), default="onecycle",
                    help="learning-rate schedule; onecycle preserves the original behavior")
    ap.add_argument("--warmup-steps", type=int, default=1000,
                    help="linear warmup steps for --sched cosine")
    ap.add_argument("--warmup-start-factor", type=float, default=0.1,
                    help="initial LR factor for cosine warmup, e.g. 0.1 means 10%% of --lr")
    ap.add_argument("--min-lr", type=float, default=1e-5,
                    help="minimum LR for --sched cosine")
    ap.add_argument("--onecycle-pct-start", type=float, default=0.05)
    ap.add_argument("--onecycle-div-factor", type=float, default=25.0)
    ap.add_argument("--onecycle-final-div-factor", type=float, default=10000.0)
    ap.add_argument("--tail-steps", type=int, default=0,
                    help="extra low-LR OneCycle steps after the main schedule, still in the same scratch run")
    ap.add_argument("--tail-lr", type=float, default=1e-5,
                    help="max LR for the optional tail OneCycle phase")
    ap.add_argument("--tail-wd", type=float, default=None,
                    help="weight decay for the optional tail phase; default keeps --wd")
    ap.add_argument("--tail-pct-start", type=float, default=0.15)
    ap.add_argument("--tail-div-factor", type=float, default=10.0)
    ap.add_argument("--tail-final-div-factor", type=float, default=10000.0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop after this many validations without > --min-delta improvement; 0 disables")
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="minimum mean angular error improvement in degrees to reset patience")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="clip global gradient norm when >0")
    ap.add_argument("--ema", action="store_true",
                    help="evaluate and save an exponential moving average of model weights")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--init-ckpt", default=None,
                    help="optional checkpoint to initialize model weights for fine-tuning")
    ap.add_argument("--eval-at-start", action="store_true",
                    help="run validation before training and seed best.pt from the initial weights")
    ap.add_argument("--save-last", action="store_true",
                    help="also save last.pt at every validation")
    ap.add_argument("--log-every", type=int, default=100,
                    help="每隔多少step写入一次训练loss/lr到train.log和metrics.csv")
    ap.add_argument("--vis-every", type=int, default=1000,
                    help="每隔多少step保存验证集可视化；0表示关闭")
    ap.add_argument("--vis-samples", type=int, default=6,
                    help="每次验证可视化误差最大的验证样本数量；0表示只画曲线/分布")
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt"))
    a = ap.parse_args()

    if a.sched == "cosine":
        if a.min_lr < 0 or a.min_lr > a.lr:
            raise ValueError("--min-lr must be in [0, --lr] for --sched cosine")
        if not 0 < a.warmup_start_factor <= 1:
            raise ValueError("--warmup-start-factor must be in (0, 1]")
    if not 0 < a.onecycle_pct_start <= 1:
        raise ValueError("--onecycle-pct-start must be in (0, 1]")
    if a.tail_steps < 0:
        raise ValueError("--tail-steps must be >= 0")
    if a.tail_steps > 0:
        if a.tail_lr <= 0:
            raise ValueError("--tail-lr must be > 0 when --tail-steps > 0")
        if not 0 < a.tail_pct_start <= 1:
            raise ValueError("--tail-pct-start must be in (0, 1]")

    os.makedirs(a.out, exist_ok=True)
    for name in ("train.log", "metrics.csv", "metrics.jsonl"):
        path = os.path.join(a.out, name)
        if os.path.exists(path):
            os.remove(path)
    dev = a.device
    history = []
    run_start = time.strftime("%Y-%m-%d %H:%M:%S")
    log_print(a.out, f"Run started: {run_start}")
    log_print(a.out, "Command: " + " ".join(sys.argv))

    # 加载标签
    L = np.load(a.labels)
    inl = L["inlier_frac"]
    agr = L["agree_deg"]

    if a.soft:
        fa, na = L["files"], L["normal"]
        w_all = (np.clip(inl, 0, 1) * np.clip(1 - agr / 30.0, 0, 1)).astype(np.float32)
        clean = np.where((inl >= a.inlier) & (agr <= a.agree))[0]
        rng = np.random.default_rng(0)
        rng.shuffle(clean)
        nval = min(300, len(clean) // 3)
        va = set(clean[:nval].tolist())
        tr = np.array([i for i in range(len(fa)) if i not in va])
        files, normals, weights = fa[tr], na[tr], w_all[tr]
        vfiles, vnormals = fa[clean[:nval]], na[clean[:nval]]
    else:
        gate = (inl >= a.inlier) & (agr <= a.agree)
        f2 = L["files"][gate]
        n2 = L["normal"][gate]
        rng = np.random.default_rng(0)
        p = rng.permutation(len(f2))
        f2, n2 = f2[p], n2[p]
        nval = max(100, len(f2) // 10)
        vfiles, vnormals = f2[:nval], n2[:nval]
        files, normals, weights = f2[nval:], n2[nval:], None

    log_print(a.out, f"MSECNet+RGB (max_points={a.max_points}): train {len(files)} / val {len(vfiles)} soft={a.soft}")
    if a.aug_deg > 0:
        log_print(a.out, "WARNING: --aug-deg > 0 rotates xyz/labels but not the camera-fixed MoGe features; "
                  "treat this as an augmentation ablation, not the default RGB-fusion setting.")

    feat_dim, cls_dim = infer_moge_feature_dims(a.moge_feat_dir, list(files) + list(vfiles))
    if a.rgb_mode in ("full", "map") and feat_dim <= 0:
        raise RuntimeError(f"rgb_mode={a.rgb_mode} needs feat_map, but no feature-map channels were found")
    if a.rgb_mode in ("full", "cls") and cls_dim <= 0:
        raise RuntimeError(f"rgb_mode={a.rgb_mode} needs cls_token, but no CLS token was found")
    rgb_feat_dim = rgb_dim_from_parts(feat_dim, cls_dim, a.rgb_mode)
    log_print(a.out, f"MoGe features: feat_dim={feat_dim} cls_dim={cls_dim} mode={a.rgb_mode} rgb_feat_dim={rgb_feat_dim}")

    missing = missing_moge_feature_files(a.moge_feat_dir, list(files) + list(vfiles))
    if missing:
        msg = f"missing {len(missing)} MoGe feature files, first missing: {missing[0]}"
        if not a.allow_missing_rgb:
            raise FileNotFoundError(msg + "; run precompute_moge_feat.py or pass --allow-missing-rgb")
        log_print(a.out, "WARNING: " + msg + "; using zero RGB descriptors for missing files")

    # 加载knob centers
    KC = json.load(open(os.path.join(ROOT, "shared", "knob_centers.json")))
    write_json(os.path.join(a.out, "run_config.json"), {
        "started_at": run_start,
        "argv": sys.argv,
        "labels": a.labels,
        "pcd_dir": a.pcd_dir,
        "moge_feat_dir": a.moge_feat_dir,
        "steps": a.steps,
        "batch_size": a.bs,
        "max_points": a.max_points,
        "inlier": a.inlier,
        "agree": a.agree,
        "radius": a.radius,
        "soft": bool(a.soft),
        "aug_deg": a.aug_deg,
        "rgb_mode": a.rgb_mode,
        "rgb_feat_dim": int(rgb_feat_dim),
        "moge_feat_dim": int(feat_dim),
        "moge_cls_dim": int(cls_dim),
        "lr": a.lr,
        "weight_decay": a.wd,
        "scheduler": a.sched,
        "warmup_steps": a.warmup_steps,
        "warmup_start_factor": a.warmup_start_factor,
        "min_lr": a.min_lr,
        "onecycle_pct_start": a.onecycle_pct_start,
        "onecycle_div_factor": a.onecycle_div_factor,
        "onecycle_final_div_factor": a.onecycle_final_div_factor,
        "tail_steps": a.tail_steps,
        "tail_lr": a.tail_lr,
        "tail_weight_decay": a.tail_wd,
        "tail_pct_start": a.tail_pct_start,
        "tail_div_factor": a.tail_div_factor,
        "tail_final_div_factor": a.tail_final_div_factor,
        "total_train_steps": int(a.steps + a.tail_steps),
        "patience": a.patience,
        "min_delta": a.min_delta,
        "grad_clip": a.grad_clip,
        "ema": bool(a.ema),
        "ema_decay": a.ema_decay,
        "init_ckpt": a.init_ckpt,
        "eval_at_start": bool(a.eval_at_start),
        "train_count": int(len(files)),
        "val_count": int(len(vfiles)),
        "val_files": [str(x) for x in vfiles],
    })

    # 创建数据集
    tr = DataLoader(
        DSWithRGB(files, normals, a.pcd_dir, a.moge_feat_dir, KC, a.radius,
                  True, a.max_points, rgb_feat_dim, weights=weights,
                  aug_deg=a.aug_deg, rgb_mode=a.rgb_mode,
                  allow_missing_rgb=a.allow_missing_rgb),
        batch_size=a.bs, shuffle=True, num_workers=8,
        drop_last=True, persistent_workers=True, collate_fn=collate
    )
    vl = DataLoader(
        DSWithRGB(vfiles, vnormals, a.pcd_dir, a.moge_feat_dir, KC, a.radius,
                  False, a.max_points, rgb_feat_dim, rgb_mode=a.rgb_mode,
                  allow_missing_rgb=a.allow_missing_rgb, return_name=True),
        batch_size=a.bs, shuffle=False, num_workers=4, collate_fn=collate
    )

    # 创建模型
    cfg = config.load_cfg_from_cfg_file(
        os.path.join(MROOT, "scripts/config/pcpnet/config.yaml")
    )
    cfg.num_classes = 3

    model = MSECNetWithRGB(cfg, rgb_feat_dim).to(dev)

    if a.init_ckpt:
        ckpt = torch.load(a.init_ckpt, map_location=dev)
        ckpt_mode = ckpt.get("rgb_mode")
        ckpt_dim = ckpt.get("rgb_feat_dim")
        if ckpt_mode is not None and ckpt_mode != a.rgb_mode:
            raise ValueError(f"--init-ckpt rgb_mode={ckpt_mode} does not match current mode={a.rgb_mode}")
        if ckpt_dim is not None and int(ckpt_dim) != int(rgb_feat_dim):
            raise ValueError(f"--init-ckpt rgb_feat_dim={ckpt_dim} does not match current dim={rgb_feat_dim}")
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=True)
        log_print(a.out, f"Initialized model weights from {a.init_ckpt}")

    ema = ModelEMA(model, a.ema_decay) if a.ema else None

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = None
    if a.sched == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=a.lr, total_steps=a.steps, pct_start=a.onecycle_pct_start,
            div_factor=a.onecycle_div_factor, final_div_factor=a.onecycle_final_div_factor
        )
    elif a.sched == "constant":
        set_optimizer_lr(opt, a.lr)

    best = 999.0
    best_step = 0
    bad_vals = 0
    it = iter(tr)
    rl = 0.0
    t0 = time.time()
    total_train_steps = int(a.steps + a.tail_steps)
    tail_sched = None

    def make_ckpt_payload(eval_model, step, mean_err, median_err, p10):
        return {
            "model": eval_model.state_dict(),
            "step": int(step),
            "mean_err": float(mean_err),
            "median_err": float(median_err),
            "p10": float(p10),
            "rgb_feat_dim": int(rgb_feat_dim),
            "rgb_mode": a.rgb_mode,
            "moge_feat_dim": int(feat_dim),
            "moge_cls_dim": int(cls_dim),
            "aug_deg": float(a.aug_deg),
            "scheduler": a.sched,
            "lr": float(a.lr),
            "min_lr": float(a.min_lr),
            "weight_decay": float(a.wd),
            "ema": bool(a.ema),
        }

    if a.eval_at_start:
        eval_model = ema.ema if ema is not None else model
        m, md, p10, details = evaluate(eval_model, vl, dev)
        best = float(m)
        best_step = 0
        record_metric(a.out, history, {
            "step": 0,
            "split": "val",
            "loss": "",
            "ema_loss": "",
            "lr": get_optimizer_lr(opt),
            "mean_ang_err": float(m),
            "median_ang_err": float(md),
            "p10": float(p10),
            "steps_per_sec": "",
            "best_mean_ang_err": float(best),
        })
        log_print(a.out, f"  [VAL step 0] mean_ang_err={m:.2f}deg median={md:.2f}deg "
                  f"<=10deg:{p10:.0f}% (initial weights)")
        write_json(os.path.join(a.out, "val_predictions", "step_000000.json"), {
            "step": 0,
            "mean_ang_err": float(m),
            "median_ang_err": float(md),
            "p10": float(p10),
            "samples": details,
        })
        write_json(os.path.join(a.out, "val_predictions", "latest.json"), {
            "step": 0,
            "mean_ang_err": float(m),
            "median_ang_err": float(md),
            "p10": float(p10),
            "samples": details,
        })
        torch.save(make_ckpt_payload(eval_model, 0, m, md, p10), os.path.join(a.out, "best.pt"))
        # Preserve the original script's effective behavior: after validation,
        # training continues with dropout/BN in eval mode unless the caller
        # explicitly changes the model mode in code.
        model.eval()

    for step in range(1, total_train_steps + 1):
        if a.tail_steps > 0 and step == a.steps + 1:
            tail_wd = a.wd if a.tail_wd is None else a.tail_wd
            for group in opt.param_groups:
                group["weight_decay"] = float(tail_wd)
            tail_sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=a.tail_lr, total_steps=a.tail_steps,
                pct_start=a.tail_pct_start, div_factor=a.tail_div_factor,
                final_div_factor=a.tail_final_div_factor
            )
            log_print(a.out, f"Starting tail phase: steps={a.tail_steps} max_lr={a.tail_lr:.3e} wd={tail_wd}")

        in_tail = a.tail_steps > 0 and step > a.steps
        if a.sched == "cosine" and not in_tail:
            set_optimizer_lr(
                opt,
                cosine_lr(
                    step, a.steps, a.lr, a.min_lr,
                    min(a.warmup_steps, max(a.steps - 1, 0)),
                    a.warmup_start_factor
                )
            )

        try:
            coord, offset, normal, w, counts, rgb_global = next(it)
        except StopIteration:
            it = iter(tr)
            coord, offset, normal, w, counts, rgb_global = next(it)

        coord = coord.to(dev)
        offset = offset.to(dev)
        normal = normal.to(dev)
        w = w.to(dev)
        rgb_global = rgb_global.to(dev)

        seg = torch.repeat_interleave(torch.arange(len(counts), device=dev), counts.to(dev))
        pp = F.normalize(
            model(coord, torch.zeros(coord.shape[0], 0, device=dev), offset, rgb_global),
            dim=1
        )

        tgt = normal[seg]
        per = 1 - (pp * tgt).sum(1) ** 2
        wpt = w[seg]
        loss = (wpt * per).sum() / (wpt.sum() + 1e-6)

        opt.zero_grad()
        loss.backward()
        if a.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        if in_tail:
            tail_sched.step()
        elif sched is not None:
            sched.step()
        if ema is not None:
            ema.update(model)
        rl = 0.9 * rl + 0.1 * loss.item()
        if a.log_every > 0 and (step % a.log_every == 0 or step == 1):
            steps_per_sec = step / max(time.time() - t0, 1e-9)
            lr_now = get_optimizer_lr(opt)
            record_metric(a.out, history, {
                "step": step,
                "split": "train",
                "loss": float(loss.item()),
                "ema_loss": float(rl),
                "lr": lr_now,
                "steps_per_sec": steps_per_sec,
                "best_mean_ang_err": best if best < 999.0 else "",
            })
            log_print(a.out, f"  [TRAIN step {step}] loss={loss.item():.5f} ema={rl:.5f} "
                      f"lr={lr_now:.3e} ({steps_per_sec:.2f}/s)")

        if step % a.val_every == 0 or step == total_train_steps:
            eval_model = ema.ema if ema is not None else model
            m, md, p10, details = evaluate(eval_model, vl, dev)
            steps_per_sec = step / max(time.time() - t0, 1e-9)
            lr_now = get_optimizer_lr(opt)
            raw_best = m < best
            meaningful_best = best >= 999.0 or m < (best - a.min_delta)
            if raw_best:
                best = m
                best_step = step
            if meaningful_best:
                bad_vals = 0
            else:
                bad_vals += 1
            record_metric(a.out, history, {
                "step": step,
                "split": "val",
                "loss": float(loss.item()),
                "ema_loss": float(rl),
                "lr": lr_now,
                "mean_ang_err": float(m),
                "median_ang_err": float(md),
                "p10": float(p10),
                "steps_per_sec": steps_per_sec,
                "best_mean_ang_err": float(best),
            })
            log_print(a.out, f"  [VAL step {step}] mean_ang_err={m:.2f}deg median={md:.2f}deg "
                      f"<=10deg:{p10:.0f}% (loss {rl:.4f}, {steps_per_sec:.1f}/s)")
            val_path = os.path.join(a.out, "val_predictions", f"step_{step:06d}.json")
            write_json(val_path, {
                "step": int(step),
                "mean_ang_err": float(m),
                "median_ang_err": float(md),
                "p10": float(p10),
                "samples": details,
            })
            write_json(os.path.join(a.out, "val_predictions", "latest.json"), {
                "step": int(step),
                "mean_ang_err": float(m),
                "median_ang_err": float(md),
                "p10": float(p10),
                "samples": details,
            })
            plot_training_curves(history, a.out)
            plot_validation_error_summary(details, step, a.out)
            if a.vis_every > 0 and (step % a.vis_every == 0 or step == total_train_steps):
                save_validation_visuals(
                    details, step, a.out, a.pcd_dir, a.moge_feat_dir, KC,
                    a.radius, a.vis_samples
                )
            ckpt_payload = make_ckpt_payload(eval_model, step, m, md, p10)
            if a.save_last:
                torch.save(ckpt_payload, os.path.join(a.out, "last.pt"))
            if raw_best:
                torch.save(ckpt_payload, os.path.join(a.out, "best.pt"))
            # When EMA is enabled, evaluation touches only ema.ema. Set the
            # trainable model to eval too so EMA runs keep the same dropout/BN
            # dynamics as the original non-EMA training script after validation.
            if ema is not None:
                model.eval()
            if a.patience > 0 and bad_vals >= a.patience:
                log_print(
                    a.out,
                    f"early stop at step {step}: no >{a.min_delta:.4f}deg val improvement "
                    f"for {bad_vals} validations; best={best:.2f}deg at step {best_step}"
                )
                break

    plot_training_curves(history, a.out)
    log_print(a.out, f"done. best mean_ang_err={best:.2f}deg at step {best_step}")


if __name__ == "__main__":
    main()
