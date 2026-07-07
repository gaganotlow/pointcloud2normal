#!/usr/bin/env python3
"""Precompute NormalNet prediction on every cloud -> normalnet/predictions.json.
Per cloud: {normal (CV, toward camera), agree (deg vs geometric label), inlier (plane-fit quality)}.
So the labeling tool can use model-assisted PRE-LABEL and SORT by quality.

Usage: python normalnet/precompute.py [checkpoint] [output.json] [--patch-radius 0.3]
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "shared"))
from train import NormalNet  # noqa: E402
import cap_patch  # noqa: E402

# ---- configurable paths ----
PCD = os.environ.get("PCD_DATASET", os.path.join(ROOT, "data", "pcd_dataset_roi"))
KC_PATH = os.path.join(ROOT, "shared", "knob_centers.json")
LABELS_PATH = os.path.join(ROOT, "shared", "normal_labels_patch03.npz")

KC = json.load(open(KC_PATH)) if os.path.exists(KC_PATH) else {}
L = np.load(LABELS_PATH) if os.path.exists(LABELS_PATH) else None
files, geom, inl = (L["files"], L["normal"], L["inlier_frac"]) if L is not None else ([], [], [])

ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "ckpt_normal", "best.pt")
OUTPATH = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "shared", "v3_predictions.json")
PATCH_RADIUS = float(sys.argv[sys.argv.index("--patch-radius") + 1]) if "--patch-radius" in sys.argv else 0.3

m = NormalNet().cuda().eval()
m.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False)["model"])
print(f"{len(files)} clouds; model {ckpt}", flush=True)

out = {}
for i, f in enumerate(files):
    f = str(f)
    kc = KC.get(f)
    if kc is None:
        continue
    try:
        d = np.load(os.path.join(PCD, f))
    except Exception:
        continue
    xyz = d["xyz"][d["label"] == 1].astype(np.float32)
    if len(xyz) < 120:
        continue
    _, pm = cap_patch.extract(xyz, d["K_norm"], int(d["w"]), int(d["h"]), kc, radius_frac=PATCH_RADIUS)
    if pm.sum() < 80:
        continue
    P = xyz[pm]; c = P.mean(0)
    Pn = (P - c) / (np.linalg.norm(P - c, axis=1).max() + 1e-9)
    idx = np.random.RandomState(0).choice(len(Pn), 1024, replace=(len(Pn) < 1024))
    x = torch.from_numpy(Pn[idx].T[None].copy()).float().cuda()
    with torch.no_grad():
        ax = m(x)[0].cpu().numpy()
    ax = ax / (np.linalg.norm(ax) + 1e-9)
    if ax @ (-c) < 0:                     # orient toward camera = outward
        ax = -ax
    if len(geom) > i:
        g = geom[i] / (np.linalg.norm(geom[i]) + 1e-9)
        agree = float(np.degrees(np.arccos(np.clip(abs(ax @ g), 0, 1))))
    else:
        agree = -1
    inl_val = round(float(inl[i]), 3) if len(inl) > i else -1
    out[f] = {"normal": [float(v) for v in ax], "agree": round(agree, 1), "inlier": inl_val}
    if (i + 1) % 1000 == 0:
        json.dump(out, open(OUTPATH, "w"))
        print(f"  {i+1}/{len(files)}  done={len(out)}", flush=True)

json.dump(out, open(OUTPATH, "w"))
ag = np.array([v["agree"] for v in out.values() if v["agree"] >= 0])
if len(ag) > 0:
    print(f"SAVED {len(out)} -> {OUTPATH}", flush=True)
    print(f"pred-vs-geom agree: median {np.median(ag):.1f}deg  <=10deg {100*(ag<=10).mean():.0f}%  >20deg {100*(ag>20).mean():.0f}%", flush=True)
