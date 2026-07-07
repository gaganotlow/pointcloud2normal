#!/usr/bin/env python3
"""MSECNet + MoGe RGB特征融合训练脚本。
将MoGe DINOv2 encoder特征投影到点云并融合到MSECNet进行法向量回归。

Usage: python train_rgb_fusion.py <labels> <pcd_dir> <moge_feat_dir> [options]
"""
import argparse
import json
import os
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
sys.path.insert(0, os.path.join(ROOT, "shared"))
from util import config
from architectures import MSECNet
import cap_patch


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
    C, feat_h, feat_w = feat_map.shape
    N = len(xyz)

    # 3D点投影到图像坐标
    xyz_homo = np.concatenate([xyz, np.ones((N, 1))], axis=1)  # (N, 4)
    uv_homo = xyz_homo[:, :3] @ K_norm.T  # (N, 3)

    # 归一化得到像素坐标
    valid = uv_homo[:, 2] > 0
    u = uv_homo[:, 0] / (uv_homo[:, 2] + 1e-9)
    v = uv_homo[:, 1] / (uv_homo[:, 2] + 1e-9)

    # 映射到特征图尺寸
    u_feat = u * feat_w / w
    v_feat = v * feat_h / h

    # 双线性插值采样
    u_feat = np.clip(u_feat, 0, feat_w - 1)
    v_feat = np.clip(v_feat, 0, feat_h - 1)

    # 简单最近邻采样（可优化为双线性插值）
    u_idx = np.round(u_feat).astype(np.int32)
    v_idx = np.round(v_feat).astype(np.int32)

    point_feats = np.zeros((N, C), dtype=np.float32)
    point_feats[valid] = feat_map[:, v_idx[valid], u_idx[valid]].T

    return point_feats


class DSWithRGB(Dataset):
    """带MoGe RGB特征的数据集。"""

    def __init__(self, files, normals, pcd_dir, moge_feat_dir, kc, radius,
                 train, max_points, weights=None, aug_deg=45.0):
        self.files = files
        self.normals = normals
        self.pcd_dir = pcd_dir
        self.moge_feat_dir = moge_feat_dir
        self.kc = kc
        self.radius = radius
        self.train = train
        self.maxp = max_points
        self.weights = weights
        self.aug_deg = aug_deg

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

        # 加载MoGe特征
        moge_feat_path = os.path.join(self.moge_feat_dir, self.files[i])
        if os.path.exists(moge_feat_path):
            moge_data = np.load(moge_feat_path)
            feat_map = moge_data["feat_map"]  # (dim_out, h_low, w_low)
            cls_token = moge_data["cls_token"]  # (1024,)

            # 投影特征到点云
            point_feats = project_features_to_points(
                xyz, feat_map, d["K_norm"], int(d["w"]), int(d["h"])
            )  # (N, dim_out)

            # 全局池化
            rgb_global = np.concatenate([
                point_feats.max(axis=0),  # max pool
                point_feats.mean(axis=0),  # avg pool
                cls_token  # CLS token
            ], axis=0).astype(np.float32)
        else:
            # 如果没有MoGe特征，用零填充
            rgb_global = np.zeros(1024 + 1024, dtype=np.float32)

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

        return xyz.astype(np.float32), n.astype(np.float32), np.float32(w), rgb_global


def collate(batch):
    coords = [torch.from_numpy(b[0]) for b in batch]
    counts = torch.tensor([len(c) for c in coords], dtype=torch.int64)
    coord = torch.cat(coords, 0).float()
    offset = torch.cumsum(counts, 0).int()
    normal = torch.from_numpy(np.stack([b[1] for b in batch])).float()
    w = torch.tensor([b[2] for b in batch]).float()
    rgb_global = torch.from_numpy(np.stack([b[3] for b in batch])).float()
    return coord, offset, normal, w, counts, rgb_global


class MSECNetWithRGB(nn.Module):
    """MSECNet + RGB特征融合头。"""

    def __init__(self, cfg, rgb_feat_dim):
        super().__init__()
        self.msecnet = MSECNet(cfg)

        # RGB特征融合头
        geom_dim = cfg.d_out_initial * (2 ** (len([s for s in cfg.strides if s > 1])))
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_feat_dim, 512),
            nn.BatchNorm1d(512),
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

        # RGB特征投影
        rgb_feat = self.rgb_proj(rgb_global)  # (B, 512)

        # 按batch展开RGB特征到每个点
        batch_size = rgb_global.shape[0]
        points_per_batch = geom_feat.shape[0] // batch_size
        rgb_feat_expanded = rgb_feat.repeat_interleave(points_per_batch, dim=0)  # (N, 512)

        # 门控融合
        alpha = self.gate(geom_feat)  # (N, 1)
        fused_feat = torch.cat([geom_feat, alpha * rgb_feat_expanded], dim=1)  # (N, geom_dim+512)

        # 最终分类
        out = self.fusion_classifier(fused_feat)
        return out


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    errs = []
    for coord, offset, normal, w, counts, rgb_global in loader:
        coord = coord.to(dev)
        offset = offset.to(dev)
        rgb_global = rgb_global.to(dev)

        pp = F.normalize(model(coord, torch.zeros(coord.shape[0], 0, device=dev),
                              offset, rgb_global), dim=1)
        idx = 0
        for b, c in enumerate(counts.tolist()):
            agg = F.normalize(pp[idx:idx + c].mean(0), dim=0)
            idx += c
            cos = float(abs(torch.dot(agg, normal[b].to(dev))).clamp(0, 1))
            errs.append(np.degrees(np.arccos(cos)))

    e = np.array(errs)
    return e.mean(), np.median(e), (e <= 10).mean() * 100


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
    ap.add_argument("--aug-deg", type=float, default=45.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt"))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"

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

    print(f"MSECNet+RGB (max_points={a.max_points}): train {len(files)} / val {len(vfiles)} soft={a.soft}",
          flush=True)

    # 加载knob centers
    KC = json.load(open(os.path.join(ROOT, "shared", "knob_centers.json")))

    # 创建数据集
    tr = DataLoader(
        DSWithRGB(files, normals, a.pcd_dir, a.moge_feat_dir, KC, a.radius,
                 True, a.max_points, weights, a.aug_deg),
        batch_size=a.bs, shuffle=True, num_workers=8,
        drop_last=True, persistent_workers=True, collate_fn=collate
    )
    vl = DataLoader(
        DSWithRGB(vfiles, vnormals, a.pcd_dir, a.moge_feat_dir, KC, a.radius,
                 False, a.max_points),
        batch_size=a.bs, shuffle=False, num_workers=4, collate_fn=collate
    )

    # 创建模型
    cfg = config.load_cfg_from_cfg_file(
        os.path.join(MROOT, "scripts/config/pcpnet/config.yaml")
    )
    cfg.num_classes = 3

    # RGB特征维度: max_pool(dim_out) + avg_pool(dim_out) + cls_token(1024)
    # 假设dim_out=768 (DINOv2 ViT-L)
    rgb_feat_dim = 768 + 768 + 1024
    model = MSECNetWithRGB(cfg, rgb_feat_dim).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05
    )

    best = 999.0
    it = iter(tr)
    rl = 0.0
    t0 = time.time()

    for step in range(1, a.steps + 1):
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
        opt.step()
        sched.step()
        rl = 0.9 * rl + 0.1 * loss.item()

        if step % a.val_every == 0 or step == a.steps:
            m, md, p10 = evaluate(model, vl, dev)
            print(f"  [VAL step {step}] mean_ang_err={m:.2f}deg median={md:.2f}deg "
                  f"<=10deg:{p10:.0f}% (loss {rl:.4f}, {step/(time.time()-t0):.1f}/s)",
                  flush=True)
            if m < best:
                best = m
                torch.save({"model": model.state_dict()},
                          os.path.join(a.out, "best.pt"))

    print(f"done. best mean_ang_err={best:.2f}deg", flush=True)


if __name__ == "__main__":
    main()
