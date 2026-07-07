#!/usr/bin/env python3
"""预计算MoGe encoder特征(DINOv2 ViT-L)用于NormalNet训练。
提取每个样本的 encoder feature map + CLS token，保存到npz文件。

Usage: python precompute_moge_feat.py <pcd_dir> <output_dir>
Output: <output_dir>/<sample_name>.npz 包含 feat_map(dim_out, h, w), cls_token(1024,)
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 导入MoGe模型
try:
    from moge.model.v2 import MoGeModel
except ImportError:
    print("请确保已安装moge并激活对应环境", file=sys.stderr)
    sys.exit(1)

WEIGHTS = "Ruicheng/moge-2-vitl-normal"


def extract_encoder_features(model, rgb_tensor, device):
    """提取MoGe encoder的中间特征。

    Args:
        model: MoGeModel实例
        rgb_tensor: (3, H, W) 归一化的RGB图像
        device: 设备

    Returns:
        feat_map: (dim_out, h_low, w_low) encoder特征图
        cls_token: (1024,) CLS token全局特征
    """
    with torch.no_grad():
        # 调用encoder，返回feature map和cls token
        image = rgb_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
        features, cls_token = model.encoder(image)

        # features: (1, dim_out, h_low, w_low)
        # cls_token: (1, 1024)
        feat_map = features[0].cpu().numpy().astype(np.float32)  # (dim_out, h_low, w_low)
        cls_tok = cls_token[0].cpu().numpy().astype(np.float32)  # (1024,)

    return feat_map, cls_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcd_dir", help="点云数据目录(包含.npz文件和对应的RGB图像)")
    ap.add_argument("output_dir", help="输出特征保存目录")
    ap.add_argument("--device", default="cuda", help="设备")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载MoGe模型
    print(f"加载MoGe模型: {WEIGHTS}", flush=True)
    model = MoGeModel.from_pretrained(WEIGHTS).to(args.device).eval()
    print(f"模型加载完成", flush=True)

    # 扫描所有.npz文件
    npz_files = sorted([f for f in os.listdir(args.pcd_dir) if f.endswith(".npz")])
    print(f"找到 {len(npz_files)} 个样本", flush=True)

    success_count = 0
    for npz_file in tqdm(npz_files, desc="提取MoGe特征"):
        npz_path = os.path.join(args.pcd_dir, npz_file)
        out_path = os.path.join(args.output_dir, npz_file)

        # 如果已存在则跳过
        if os.path.exists(out_path):
            success_count += 1
            continue

        try:
            # 加载npz数据
            d = np.load(npz_path)

            # 检查是否有rgb字段
            if "rgb" not in d:
                continue

            rgb = d["rgb"]  # (H, W, 3)
            h, w = rgb.shape[:2]

            # 转换为tensor并归一化
            rgb_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0  # (3, H, W)

            # 提取encoder特征
            feat_map, cls_token = extract_encoder_features(model, rgb_tensor, args.device)

            # 保存特征
            np.savez_compressed(
                out_path,
                feat_map=feat_map,      # (dim_out, h_low, w_low)
                cls_token=cls_token,    # (1024,)
                h=h,
                w=w
            )

            success_count += 1

        except Exception as e:
            print(f"处理 {npz_file} 时出错: {e}", file=sys.stderr)
            continue

    print(f"\n完成! 成功提取 {success_count}/{len(npz_files)} 个样本的特征", flush=True)
    print(f"特征保存至: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
