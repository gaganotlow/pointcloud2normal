#!/usr/bin/env python3
"""Persistent MoGe worker (env 02_depth_moge2): load MoGe ONCE, process queued full-image requests.

The labeling server drops <queue>/<key>.req (json {img, txt, out}); this worker runs MoGe on the FULL
image (no voxel downsample — full density), rasterizes the manual inner_cover polygon (cls 1) from the
YOLO-seg .txt, and writes the cache npz {xyz, rgb, label, K_norm, w, h}. ~2-4 s/image (model stays loaded).
Run: HF_HOME=$PWD/models/hf_cache CUDA_VISIBLE_DEVICES=<g> python moge_worker.py
"""
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = "/tmp/fuelcap_fcc"                        # LOCAL ssd, not NFS -> fast np.load
QDIR = os.path.join(CACHE, "_queue")
os.makedirs(QDIR, exist_ok=True)
CTX_BUDGET = 120000      # keep ALL inner_cover points (full density); cap background context (smaller npz=faster load)

model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda").eval()
print("MoGe worker ready", flush=True)


def process(img, txt, out):
    bgr = cv2.imread(img)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    t = (torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0).cuda()
    with torch.no_grad():
        o = model.infer(t)
    pts = o["points"].cpu().numpy().astype(np.float32)
    valid = o["mask"].cpu().numpy().astype(bool) & np.isfinite(pts).all(-1)
    K = o["intrinsics"].cpu().numpy().astype(np.float32)
    mask = np.zeros((H, W), np.uint8)
    if txt and os.path.exists(txt):
        for line in open(txt).read().strip().split("\n"):
            p = line.split()
            if len(p) >= 7 and p[0] == "1":                       # cls 1 = inner_cover
                poly = (np.array(p[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)
    lab = mask.astype(bool)
    vp, vc, vl = pts[valid], rgb[valid], lab[valid].astype(np.uint8)
    inner_idx = np.where(vl == 1)[0]                           # keep ALL cap points (full density)
    ctx_idx = np.where(vl == 0)[0]
    if len(ctx_idx) > CTX_BUDGET:
        ctx_idx = np.random.RandomState(0).choice(ctx_idx, CTX_BUDGET, replace=False)
    keep = np.concatenate([inner_idx, ctx_idx])
    vp, vc, vl = vp[keep], vc[keep], vl[keep]
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, xyz=vp, rgb=vc, label=vl, K_norm=K, w=W, h=H)
    os.replace(tmp, out)
    print(f"  built {os.path.basename(out)[:40]} pts={len(vp)} inner={int(vl.sum())}", flush=True)


while True:
    reqs = sorted(glob.glob(os.path.join(QDIR, "*.req")))          # user loads = priority
    if not reqs:
        reqs = sorted(glob.glob(os.path.join(QDIR, "*.pre")))      # prefetch = lower priority
    if not reqs:
        time.sleep(0.12); continue
    r = reqs[0]
    try:
        j = json.load(open(r))
    except Exception:
        os.remove(r); continue
    try:
        if not os.path.exists(j["out"]):                          # skip if already cached
            process(j["img"], j.get("txt"), j["out"])
    except Exception as e:
        open(j["out"] + ".err", "w").write(str(e)[:300])
        print("  ERR", str(e)[:150], flush=True)
    finally:
        if os.path.exists(r):
            os.remove(r)
