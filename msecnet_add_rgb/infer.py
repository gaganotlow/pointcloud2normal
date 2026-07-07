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
from train_rgb_fusion import MSECNetWithRGB, project_features_to_points
import cap_patch


def load_sample(npz_file, pcd_dir, moge_feat_dir, kc_dict, radius=0.3):
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

    # 加载MoGe特征
    moge_feat_path = os.path.join(moge_feat_dir, npz_file)
    if os.path.exists(moge_feat_path):
        moge_data = np.load(moge_feat_path)
        feat_map = moge_data["feat_map"]
        cls_token = moge_data["cls_token"]

        # 投影特征
        point_feats = project_features_to_points(
            xyz, feat_map, d["K_norm"], int(d["w"]), int(d["h"])
        )

        # 全局池化
        rgb_global = np.concatenate([
            point_feats.max(axis=0),
            point_feats.mean(axis=0),
            cls_token
        ], axis=0).astype(np.float32)
    else:
        rgb_global = np.zeros(1024 + 1024, dtype=np.float32)

    # 中心化和归一化
    xyz = xyz - xyz.mean(0)
    xyz = xyz / (np.linalg.norm(xyz, axis=1).max() + 1e-9)

    return xyz, rgb_global


@torch.no_grad()
def predict(model, xyz, rgb_global, device):
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
    normal = F.normalize(pp.mean(0), dim=0).cpu().numpy()

    return normal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="训练好的模型检查点")
    ap.add_argument("pcd_dir", help="点云数据目录")
    ap.add_argument("moge_feat_dir", help="MoGe特征目录")
    ap.add_argument("output_json", help="输出预测结果JSON")
    ap.add_argument("--device", default="cuda", help="设备")
    ap.add_argument("--radius", type=float, default=0.3, help="Knob patch半径")
    args = ap.parse_args()

    # 加载模型
    print(f"加载模型: {args.checkpoint}", flush=True)
    cfg = config.load_cfg_from_cfg_file(
        os.path.join(MROOT, "scripts/config/pcpnet/config.yaml")
    )
    cfg.num_classes = 3
    rgb_feat_dim = 768 + 768 + 1024
    model = MSECNetWithRGB(cfg, rgb_feat_dim).to(args.device)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("模型加载完成", flush=True)

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
            xyz, rgb_global = load_sample(
                npz_file, args.pcd_dir, args.moge_feat_dir, KC, args.radius
            )
            normal = predict(model, xyz, rgb_global, args.device)
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
