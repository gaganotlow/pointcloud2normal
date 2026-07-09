#!/usr/bin/env python3
"""End-to-end MSECNet + CNN-RGB inference.

Input is one RGB image and one point-cloud npz. No offline RGB feature
directory is needed.
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
sys.path.insert(0, HERE)

from train_cnn_rgb import (  # noqa: E402
    MSECNetWithCNNRGB,
    image_to_tensor,
    project_points_to_image_norm,
)
from util import config  # noqa: E402
import cap_patch  # noqa: E402


def read_rgb(path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 is required to read RGB images") from exc
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_knob_center(value):
    if not value:
        return None
    parts = value.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("--knob-center must be 'x,y' or 'x y'")
    return [float(parts[0]), float(parts[1])]


def load_cloud_patch(npz_path, knob_center=None, radius=0.3):
    d = np.load(npz_path)
    if "label" in d:
        xyz = d["xyz"][d["label"] == 1].astype(np.float32)
    else:
        xyz = d["xyz"].astype(np.float32)
    if len(xyz) == 0:
        xyz = d["xyz"].astype(np.float32)

    if knob_center is not None and len(xyz) >= 120:
        _, pm = cap_patch.extract(
            xyz, d["K_norm"], int(d["w"]), int(d["h"]),
            knob_center, radius_frac=radius,
        )
        if pm.sum() >= 80:
            xyz = xyz[pm]
    return d, xyz


@torch.no_grad()
def predict(model, xyz, image_tensor, grid, device):
    centroid = xyz.mean(0).astype(np.float32)
    xyz_norm = xyz - centroid
    xyz_norm = xyz_norm / (np.linalg.norm(xyz_norm, axis=1).max() + 1e-9)

    coord = torch.from_numpy(xyz_norm.astype(np.float32)).float().to(device)
    offset = torch.tensor([len(xyz_norm)], dtype=torch.int32, device=device)
    image = image_tensor.unsqueeze(0).float().to(device)
    grid = torch.from_numpy(grid).float().to(device)

    pp = model(coord, torch.zeros(coord.shape[0], 0, device=device), offset, image, grid)
    pp = F.normalize(pp, dim=1).cpu().numpy()

    camdir = -centroid / (np.linalg.norm(centroid) + 1e-9)
    flip = (pp @ camdir) < 0
    pp[flip] = -pp[flip]
    normal = pp.mean(0)
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    if normal @ camdir < 0:
        normal = -normal
    return normal.astype(np.float32), centroid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("rgb_image")
    ap.add_argument("cloud_npz")
    ap.add_argument("output_json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--knob-center", default=None,
                    help="optional image-space knob center, e.g. '320,240'")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    image_size = int(ckpt.get("image_size", 256))
    cnn_dim = int(ckpt.get("cnn_dim", 128))
    rgb_feat_dim = int(ckpt.get("rgb_feat_dim", 512))
    rgb_backbone = str(ckpt.get("rgb_backbone", "light"))
    resnet_stage = str(ckpt.get("resnet_stage", "layer2"))

    cfg = config.load_cfg_from_cfg_file(os.path.join(MROOT, "scripts/config/pcpnet/config.yaml"))
    cfg.num_classes = 3
    model = MSECNetWithCNNRGB(
        cfg,
        cnn_dim=cnn_dim,
        rgb_feat_dim=rgb_feat_dim,
        rgb_backbone=rgb_backbone,
        rgb_pretrained=False,
        resnet_stage=resnet_stage,
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    knob_center = parse_knob_center(args.knob_center)
    cloud, xyz = load_cloud_patch(args.cloud_npz, knob_center, args.radius)
    rgb = read_rgb(args.rgb_image)
    image_tensor = image_to_tensor(rgb, image_size)
    grid = project_points_to_image_norm(
        xyz, cloud["K_norm"], int(cloud["w"]), int(cloud["h"]), image_size
    )
    normal, centroid = predict(model, xyz, image_tensor, grid, args.device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({
            "normal": [float(v) for v in normal],
            "centroid": [float(v) for v in centroid],
            "points": int(len(xyz)),
            "checkpoint": args.checkpoint,
            "rgb_image": args.rgb_image,
            "cloud_npz": args.cloud_npz,
            "image_size": image_size,
            "rgb_backbone": rgb_backbone,
            "resnet_stage": resnet_stage,
        }, f, indent=2, ensure_ascii=False)
    print(f"normal={np.round(normal, 4).tolist()} -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
