#!/usr/bin/env python3
"""Zero-training MoGe normal aggregate: use MoGe's per-point normal map inside the knob patch
-> one cap normal per cloud. Pure numpy, no GPU/net needed (MoGe normals already in the .npz).

Usage: python shared/precompute_moge_normal.py [output.json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cap_patch  # noqa

PCD = os.environ.get("PCD_DATASET", os.path.join(ROOT, "data", "pcd_dataset_roi"))
KC = json.load(open(os.path.join(HERE, "knob_centers.json")))
L = np.load(os.path.join(HERE, "normal_labels_patch03.npz"))
files, geom = L["files"], L["normal"]
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "moge_norm_predictions.json")

out = {}
for i, f in enumerate(files):
    f = str(f); kc = KC.get(f)
    if kc is None:
        continue
    try:
        d = np.load(os.path.join(PCD, f))
    except Exception:
        continue
    if "normal" not in d:
        continue
    inner = d["label"] == 1; xyz = d["xyz"][inner].astype(np.float32); nm = d["normal"][inner].astype(np.float32)
    if len(xyz) < 120:
        continue
    _, pm = cap_patch.extract(xyz, d["K_norm"], int(d["w"]), int(d["h"]), kc, radius_frac=0.3)
    if pm.sum() < 80:
        continue
    P = xyz[pm]; NP = nm[pm]; c = P.mean(0); cam = -c / (np.linalg.norm(c) + 1e-9)
    NP = NP / (np.linalg.norm(NP, axis=1, keepdims=True) + 1e-9)
    fl = (NP @ cam) < 0; NP[fl] = -NP[fl]                 # orient each MoGe normal toward camera
    ax = NP.mean(0); ax = ax / (np.linalg.norm(ax) + 1e-9)
    if ax @ cam < 0:
        ax = -ax
    g = geom[i] / (np.linalg.norm(geom[i]) + 1e-9)
    agree = float(np.degrees(np.arccos(np.clip(abs(ax @ g), 0, 1))))
    out[f] = {"normal": [float(v) for v in ax], "agree": round(agree, 1)}
    if (i + 1) % 2000 == 0:
        json.dump(out, open(OUT, "w")); print(f"  {i+1}/{len(files)} done={len(out)}", flush=True)

json.dump(out, open(OUT, "w"))
ag = np.array([v["agree"] for v in out.values()])
print(f"SAVED {len(out)} -> {OUT} | MoGe-vs-geom median {np.median(ag):.1f}deg", flush=True)
