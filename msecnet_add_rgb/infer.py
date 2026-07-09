#!/usr/bin/env python3
"""使用训练好的MSECNet+RGB模型进行推理。

Usage: python infer.py <checkpoint> <pcd_dir> <moge_feat_dir> <output_json>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MROOT = os.path.join(HERE, "..", "msecnet", "MSECNet")
sys.path.insert(0, os.path.join(MROOT, "model"))
sys.path.insert(0, os.path.join(MROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "shared"))

from util import config
from train_rgb_fusion import (
    MSECNetWithRGB,
    build_rgb_global,
    infer_moge_feature_dims,
    rgb_dim_from_parts,
)
import cap_patch


def load_sample(npz_file, pcd_dir, moge_feat_dir, kc_dict, rgb_feat_dim, rgb_mode,
                radius=0.3, allow_missing_rgb=False):
    """加载单个样本的数据。"""
    # 加载点云
    d = np.load(os.path.join(pcd_dir, npz_file))
    xyz = d["xyz"][d["label"] == 1].astype(np.float32)

    # 提取knob patch
    kc = kc_dict.get(npz_file)
    if kc is not None and len(xyz) >= 120:
        _, pm = cap_patch.extract(xyz, d["K_norm"], int(d["w"]), int(d["h"]),
                                 kc, radius_frac=radius)
        if pm.sum() >= 80:
            xyz = xyz[pm]

    if len(xyz) == 0:
        xyz = d["xyz"].astype(np.float32)

    moge_feat_path = os.path.join(moge_feat_dir, npz_file)
    rgb_global = build_rgb_global(xyz, d, moge_feat_path, rgb_feat_dim, rgb_mode, allow_missing_rgb)

    # 中心化和归一化
    centroid = xyz.mean(0)
    xyz = xyz - centroid
    xyz = xyz / (np.linalg.norm(xyz, axis=1).max() + 1e-9)

    return xyz, rgb_global, centroid.astype(np.float32)


@torch.no_grad()
def predict(model, xyz, rgb_global, centroid, device):
    """预测单个样本的法向量。"""
    model.eval()

    # 转换为tensor
    coord = torch.from_numpy(xyz).float().to(device)
    offset = torch.tensor([len(xyz)], dtype=torch.int32).to(device)
    rgb_global = torch.from_numpy(rgb_global).float().unsqueeze(0).to(device)

    # 前向推理
    pp = model(coord, torch.zeros(coord.shape[0], 0, device=device), offset, rgb_global)
    pp = F.normalize(pp, dim=1)

    # 聚合per-point预测
    pp_np = pp.cpu().numpy()
    camdir = -centroid / (np.linalg.norm(centroid) + 1e-9)
    flip = (pp_np @ camdir) < 0
    pp_np[flip] = -pp_np[flip]
    normal = pp_np.mean(0)
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    if normal @ camdir < 0:
        normal = -normal

    return normal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="训练好的模型检查点")
    ap.add_argument("pcd_dir", help="点云数据目录")
    ap.add_argument("moge_feat_dir", help="MoGe特征目录")
    ap.add_argument("output_json", help="输出预测结果JSON")
    ap.add_argument("--device", default="cuda", help="设备")
    ap.add_argument("--radius", type=float, default=0.3, help="Knob patch半径")
    ap.add_argument("--rgb-mode", choices=("full", "map", "cls"), default=None,
                    help="默认读取checkpoint中的rgb_mode；旧checkpoint可用此参数指定")
    ap.add_argument("--allow-missing-rgb", action="store_true",
                    help="允许缺失MoGe特征文件并以零向量兜底；默认跳过缺失样本")
    args = ap.parse_args()

    # 加载模型
    print(f"加载模型: {args.checkpoint}", flush=True)
    cfg = config.load_cfg_from_cfg_file(
        os.path.join(MROOT, "scripts/config/pcpnet/config.yaml")
    )
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg.num_classes = 3
    rgb_mode = args.rgb_mode or ckpt.get("rgb_mode", "full")
    if "rgb_feat_dim" in ckpt:
        rgb_feat_dim = int(ckpt["rgb_feat_dim"])
    else:
        feat_dim, cls_dim = infer_moge_feature_dims(args.moge_feat_dir)
        rgb_feat_dim = rgb_dim_from_parts(feat_dim, cls_dim, rgb_mode)
    model = MSECNetWithRGB(cfg, rgb_feat_dim).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"模型加载完成 rgb_mode={rgb_mode} rgb_feat_dim={rgb_feat_dim}", flush=True)

    # 加载knob centers
    kc_path = os.path.join(ROOT, "shared", "knob_centers.json")
    KC = json.load(open(kc_path)) if os.path.exists(kc_path) else {}

    # 扫描所有样本
    npz_files = sorted([f for f in os.listdir(args.pcd_dir) if f.endswith(".npz")])
    print(f"找到 {len(npz_files)} 个样本", flush=True)

    results = {}
    for i, npz_file in enumerate(npz_files):
        if (i + 1) % 100 == 0:
            print(f"  处理 {i+1}/{len(npz_files)}...", flush=True)

        try:
            xyz, rgb_global, centroid = load_sample(
                npz_file, args.pcd_dir, args.moge_feat_dir, KC,
                rgb_feat_dim, rgb_mode, args.radius, args.allow_missing_rgb
            )
            normal = predict(model, xyz, rgb_global, centroid, args.device)
            results[npz_file] = {
                "normal": [float(v) for v in normal]
            }
        except Exception as e:
            print(f"处理 {npz_file} 时出错: {e}", file=sys.stderr)
            continue

    # 保存结果
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n完成! 预测了 {len(results)}/{len(npz_files)} 个样本", flush=True)
    print(f"结果保存至: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
