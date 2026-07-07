#!/usr/bin/env python3
"""Stage 2 (env: components/02_depth_moge2): RGB -> MoGe-2 metric point map.

Usage: python s2_moge.py <image> <workdir>
Writes: <workdir>/moge.npz  (points HxWx3, valid HxW, rgb HxWx3, normal HxWx3, K_norm 3x3)
        <workdir>/cloud_full.ply
"""
import os
import sys

import cv2
import numpy as np
import torch
import trimesh
from moge.model.v2 import MoGeModel

WEIGHTS = "Ruicheng/moge-2-vitl-normal"


def main():
    image, workdir = sys.argv[1], sys.argv[2]
    os.makedirs(workdir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(WEIGHTS).to(dev).eval()
    bgr = cv2.imread(image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    t = (torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0).to(dev)
    with torch.no_grad():
        out = model.infer(t)
    points = out["points"].cpu().numpy().astype(np.float32)
    valid = out["mask"].cpu().numpy().astype(bool) & np.isfinite(out["points"].cpu().numpy()).all(-1)
    K = out["intrinsics"].cpu().numpy().astype(np.float32)
    normal = out["normal"].cpu().numpy().astype(np.float32) if out.get("normal") is not None else np.zeros_like(points)
    np.savez_compressed(os.path.join(workdir, "moge.npz"),
                        points=points, valid=valid, rgb=rgb, normal=normal, K_norm=K, w=w, h=h)
    vp, vc = points[valid], rgb[valid]
    trimesh.PointCloud(vp, colors=vc).export(os.path.join(workdir, "cloud_full.ply"))
    bb = vp.max(0) - vp.min(0)
    print(f"[s2] points={points.shape} valid={valid.sum()}/{h*w} "
          f"depthMed={np.median(vp[:,2]):.3f} bbox=({bb[0]:.3f},{bb[1]:.3f},{bb[2]:.3f}) "
          f"K_norm fx={K[0,0]:.3f} fy={K[1,1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
