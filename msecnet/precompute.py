#!/usr/bin/env python3
"""Precompute MSECNet predictions over every cloud -> pose/msecnet_predictions.json.
Per cloud: run MSECNet (per-point normals) on the 0.3 patch, orient each toward camera, aggregate
(mean->normalize) into ONE cap normal. Env: components/05_msecnet/.venv. GPU."""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MROOT = os.path.join(HERE, "MSECNet")
sys.path.insert(0, os.path.join(MROOT, "model"))
sys.path.insert(0, os.path.join(MROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "shared"))
from util import config            # noqa: E402
from architectures import MSECNet  # noqa: E402
import cap_patch                   # noqa: E402

PCD = os.environ.get("PCD_DATASET", os.path.join(ROOT, "data", "pcd_dataset_roi"))
KC_PATH = os.path.join(ROOT, "shared", "knob_centers.json")
LABELS_PATH = os.path.join(ROOT, "shared", "normal_labels_patch03.npz")
KC = json.load(open(KC_PATH)) if os.path.exists(KC_PATH) else {}
L = np.load(LABELS_PATH) if os.path.exists(LABELS_PATH) else None
files = L["files"] if L is not None else []
geom = L["normal"] if L is not None else []
inl = L["inlier_frac"] if L is not None else []
ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "ckpt_msecnet", "best.pt")
OUTPATH = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "shared", "msecnet_predictions.json")

cfg = config.load_cfg_from_cfg_file(os.path.join(MROOT, "scripts/config/pcpnet/config.yaml"))
cfg.num_classes = 3
m = MSECNet(cfg).cuda().eval()
m.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False)["model"])
print(f"{len(files)} clouds; MSECNet {ckpt}", flush=True)

out = {}
for i, f in enumerate(files):
    f = str(f); kc = KC.get(f)
    if kc is None:
        continue
    try:
        d = np.load(os.path.join(PCD, f))
    except Exception:
        continue
    xyz = d["xyz"][d["label"] == 1].astype(np.float32)
    if len(xyz) < 120:
        continue
    _, pm = cap_patch.extract(xyz, d["K_norm"], int(d["w"]), int(d["h"]), kc, radius_frac=0.3)
    if pm.sum() < 80:
        continue
    P = xyz[pm]; c = P.mean(0)
    Pn = (P - c) / (np.linalg.norm(P - c, axis=1).max() + 1e-9)
    idx = np.random.RandomState(0).choice(len(Pn), 1024, replace=(len(Pn) < 1024))
    coord = torch.from_numpy(Pn[idx].copy()).float().cuda()      # (1024,3)
    feat = torch.zeros(1024, 0).cuda()
    offset = torch.tensor([1024], dtype=torch.int32).cuda()
    with torch.no_grad():
        pp = F.normalize(m(coord, feat, offset), dim=1).cpu().numpy()   # (1024,3) per-point
    # orient each per-point normal toward camera (toward -c) for consistent aggregation
    camdir = -c / (np.linalg.norm(c) + 1e-9)
    flip = (pp @ camdir) < 0
    pp[flip] = -pp[flip]
    ax = pp.mean(0); ax = ax / (np.linalg.norm(ax) + 1e-9)
    if ax @ camdir < 0:
        ax = -ax
    if i < len(geom):
        g = geom[i] / (np.linalg.norm(geom[i]) + 1e-9)
        agree = float(np.degrees(np.arccos(np.clip(abs(ax @ g), 0, 1))))
    else:
        agree = -1
    inl_val = round(float(inl[i]), 3) if i < len(inl) else -1
    out[f] = {"normal": [float(v) for v in ax], "agree": round(agree, 1), "inlier": inl_val}
    if (i + 1) % 1000 == 0:
        json.dump(out, open(OUTPATH, "w"))
        print(f"  {i+1}/{len(files)} done={len(out)}", flush=True)

json.dump(out, open(OUTPATH, "w"))
ag = np.array([v["agree"] for v in out.values() if v["agree"] >= 0])
if len(ag) > 0:
    print(f"SAVED {len(out)} -> {OUTPATH}", flush=True)
    print(f"MSECNet-vs-geom agree: median {np.median(ag):.1f}deg  <=10deg {100*(ag<=10).mean():.0f}%", flush=True)
