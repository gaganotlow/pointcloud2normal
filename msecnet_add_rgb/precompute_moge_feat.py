#!/usr/bin/env python3
"""预计算MoGe encoder特征(DINOv2 ViT-L)用于NormalNet训练。
提取每个样本的 encoder feature map + CLS token，保存到npz文件。

Usage: python precompute_moge_feat.py <pcd_dir> <output_dir>
Output: <output_dir>/<sample_name>.npz 包含 feat_map(dim_out, h, w), cls_token(1024,)
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WEIGHTS = "Ruicheng/moge-2-vitl-normal"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def load_rgb_image(npz_path, npz_data, image_dir=None):
    """Load an HxWx3 RGB image from the npz or a same-stem file in image_dir."""
    if "rgb" in npz_data:
        rgb = npz_data["rgb"]
        if rgb.ndim == 3 and rgb.shape[-1] == 3:
            return rgb

    if image_dir:
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        candidates = [os.path.join(image_dir, stem)]
        if "__" in stem:
            car, image_stem = stem.split("__", 1)
            candidates.append(os.path.join(image_dir, car, image_stem))
        for ext in IMAGE_EXTS:
            for prefix in candidates:
                img_path = prefix + ext
                if not os.path.exists(img_path):
                    continue
                try:
                    import cv2
                except ImportError as exc:
                    raise RuntimeError("--image-dir requires cv2 to read image files") from exc
                bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"failed to read image: {img_path}")
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raise ValueError(
        f"{os.path.basename(npz_path)} does not contain full-image rgb (H,W,3). "
        "If the cached rgb is per-point (N,3), pass --image-dir with same-stem images."
    )


def extract_encoder_features(model, rgb_tensor, device, num_tokens=None, resolution_level=9):
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
        image = rgb_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
        image = image.to(dtype=model.dtype, device=model.device)

        _, _, img_h, img_w = image.shape
        aspect_ratio = img_w / img_h
        if num_tokens is None:
            min_tokens, max_tokens = model.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
        base_h = round((num_tokens / aspect_ratio) ** 0.5)
        base_w = round((num_tokens * aspect_ratio) ** 0.5)

        # MoGe v2 encoder needs the token grid size and returns the class token on request.
        features, cls_token = model.encoder(image, base_h, base_w, return_class_token=True)

        if features.ndim != 4:
            raise ValueError(f"expected encoder feature map with shape (B,C,H,W), got {tuple(features.shape)}")
        if cls_token.ndim != 2:
            raise ValueError(f"expected CLS token with shape (B,C), got {tuple(cls_token.shape)}")

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
    ap.add_argument("--image-dir", default=None,
                    help="可选：原始RGB图像目录；当npz里的rgb是(N,3)点颜色而不是(H,W,3)图像时必须提供")
    ap.add_argument("--num-tokens", type=int, default=None,
                    help="MoGe/DINO encoder token数；默认沿用MoGe resolution_level")
    ap.add_argument("--resolution-level", type=int, default=9,
                    help="未指定--num-tokens时使用，默认9，对应MoGe最高默认token数")
    ap.add_argument("--limit", type=int, default=0, help="调试用：只处理前N个样本，0表示全量")
    ap.add_argument("--save-dtype", choices=("float16", "float32"), default="float16",
                    help="保存feat_map/cls_token的数据类型；默认float16以降低磁盘占用")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的特征文件")
    ap.add_argument("--num-shards", type=int, default=1, help="并行分片总数")
    ap.add_argument("--shard-index", type=int, default=0, help="当前分片编号，范围[0,num_shards)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        from moge.model.v2 import MoGeModel
    except ImportError:
        print("请确保已安装moge并激活对应环境", file=sys.stderr)
        sys.exit(1)

    # 加载MoGe模型
    print(f"加载MoGe模型: {WEIGHTS}", flush=True)
    model = MoGeModel.from_pretrained(WEIGHTS).to(args.device).eval()
    print(f"模型加载完成", flush=True)

    # 扫描所有.npz文件
    npz_files = sorted([f for f in os.listdir(args.pcd_dir) if f.endswith(".npz")])
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        npz_files = npz_files[args.shard_index::args.num_shards]
    if args.limit > 0:
        npz_files = npz_files[:args.limit]
    shard_msg = f" shard {args.shard_index}/{args.num_shards}" if args.num_shards > 1 else ""
    print(f"找到 {len(npz_files)} 个样本{shard_msg}", flush=True)

    success_count = 0
    for npz_file in tqdm(npz_files, desc="提取MoGe特征"):
        npz_path = os.path.join(args.pcd_dir, npz_file)
        out_path = os.path.join(args.output_dir, npz_file)

        # 如果已存在则跳过
        if os.path.exists(out_path) and not args.overwrite:
            success_count += 1
            continue

        tmp_path = None
        try:
            # 加载npz数据
            d = np.load(npz_path)

            rgb = load_rgb_image(npz_path, d, args.image_dir)
            h, w = rgb.shape[:2]

            rgb = np.asarray(rgb)
            rgb_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1)
            if rgb_tensor.max() > 1.5:
                rgb_tensor = rgb_tensor / 255.0

            # 提取encoder特征
            feat_map, cls_token = extract_encoder_features(
                model, rgb_tensor, args.device, args.num_tokens, args.resolution_level
            )
            if args.save_dtype == "float16":
                feat_map = feat_map.astype(np.float16)
                cls_token = cls_token.astype(np.float16)
            else:
                feat_map = feat_map.astype(np.float32)
                cls_token = cls_token.astype(np.float32)

            tmp_path = f"{out_path}.tmp.{os.getpid()}.npz"
            np.savez_compressed(
                tmp_path,
                feat_map=feat_map,      # (dim_out, h_low, w_low)
                cls_token=cls_token,    # (1024,)
                h=h,
                w=w,
                feat_dim=np.array(feat_map.shape[0], dtype=np.int32),
                cls_dim=np.array(cls_token.shape[0], dtype=np.int32),
                weights=np.array(WEIGHTS),
                save_dtype=np.array(args.save_dtype)
            )
            os.replace(tmp_path, out_path)

            success_count += 1

        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"处理 {npz_file} 时出错: {e}", file=sys.stderr)
            continue

    print(f"\n完成! 成功提取 {success_count}/{len(npz_files)} 个样本的特征", flush=True)
    print(f"特征保存至: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
