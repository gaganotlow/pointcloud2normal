#!/usr/bin/env python3
"""Stage C (env 03_pcdseg): NormalNet inference + orient toward camera + draw arrow.
Usage: _norminfer.py <workdir> <image> <ckpt>"""
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "shared"))
from train import NormalNet  # noqa: E402

wd, img_path, ckpt = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.load(wd + "/moge.npz")
points, valid, K = z["points"], z["valid"].astype(bool), z["K_norm"]
W, H = int(z["w"]), int(z["h"])
mask = cv2.imread(wd + "/mask_inner.png", 0) > 127
if mask.shape != (H, W):
    mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), 1) > 0
sel = mask & valid & np.isfinite(points).all(-1)
P = points[sel].astype(np.float32)
if len(P) < 100:
    print("too few inner_cover points"); sys.exit(0)
# option 2: local cap-face patch around the knob center (anchor = knob 3D surface point)
import cap_patch  # noqa: E402
seg = json.load(open(wd + "/segobb.json")) if os.path.exists(wd + "/segobb.json") else {}
kc = seg.get("kc")
if kc is not None:
    knob3d, pm = cap_patch.extract(P, K, W, H, kc, radius_frac=0.5)
    if pm.sum() >= 80:
        P = P[pm]
centroid = P.mean(0)
Pn = P - centroid
Pn = Pn / (np.linalg.norm(Pn, axis=1).max() + 1e-9)
idx = np.random.RandomState(0).choice(len(Pn), 1024, replace=(len(Pn) < 1024))
x = torch.from_numpy(Pn[idx].T[None].copy()).float().cuda()
m = NormalNet().cuda().eval()
m.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False)["model"])
with torch.no_grad():
    axis = m(x)[0].cpu().numpy()
axis = axis / (np.linalg.norm(axis) + 1e-9)
if axis @ (-centroid) < 0:          # orient toward camera (origin) = outward
    axis = -axis
diam = float(np.linalg.norm(P.max(0) - P.min(0)))
json.dump({"normal": axis.tolist(), "centroid": centroid.tolist()}, open(wd + "/normal.json", "w"))

fx, fy, cx, cy = K[0, 0] * W, K[1, 1] * H, K[0, 2] * W, K[1, 2] * H
im = cv2.imread(img_path)

def proj(Pt):
    return (int(fx * Pt[0] / Pt[2] + cx), int(fy * Pt[1] / Pt[2] + cy))

o = proj(centroid); tip = proj(centroid + axis * 0.6 * diam)
cv2.arrowedLine(im, o, tip, (0, 0, 255), 3, tipLength=0.3)
cv2.circle(im, o, 5, (0, 255, 255), -1)
cv2.putText(im, f"pred normal=({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f})", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
out = os.path.join(ROOT, "output", "normal_pred_demo")
os.makedirs(out, exist_ok=True)
cv2.imwrite(out + "/" + os.path.basename(img_path).split(".")[0] + "__pred.jpg", im)
print(f"normal={np.round(axis,3).tolist()} -> saved")
