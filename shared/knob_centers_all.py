#!/usr/bin/env python3
"""Batch OBB knob centers for ALL pcd clouds -> shared/knob_centers.json (frac coords).

Requires: YOLO OBB model (ultralytics).
Set MODEL_PATH env var or edit the path below to point to inner_obb_clean_v11m_0129/best.pt.
Set SRC_DIR env var or edit below to point to the source images directory.

Usage: python knob_centers_all.py [--pcd-dir <dir>] [--out <path>]
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---- configurable paths ----
PCD = os.environ.get("PCD_DATASET", os.path.join(ROOT, "data", "pcd_dataset_roi"))
OUTJ = os.environ.get("KNOB_OUT", os.path.join(HERE, "knob_centers.json"))
SRC_DIR = os.environ.get("SRC_DIR", os.path.join(ROOT, "data", "yolo_seg_by_car"))
MODEL_PATH = os.environ.get("OBB_MODEL",
    os.path.join(ROOT, "models", "inner_obb_clean_v11m_0129", "best.pt"))

# allow CLI overrides
if "--pcd-dir" in sys.argv:
    PCD = sys.argv[sys.argv.index("--pcd-dir") + 1]
if "--out" in sys.argv:
    OUTJ = sys.argv[sys.argv.index("--out") + 1]
if "--src-dir" in sys.argv:
    SRC_DIR = sys.argv[sys.argv.index("--src-dir") + 1]
if "--model" in sys.argv:
    MODEL_PATH = sys.argv[sys.argv.index("--model") + 1]

SRC = {}
for p in glob.glob(os.path.join(SRC_DIR, "**", "*.png"), recursive=True):
    SRC[os.path.splitext(os.path.basename(p))[0]] = p


def srcpath(stem):
    return SRC.get(stem) or SRC.get(stem.split("__", 1)[-1])


m = YOLO(MODEL_PATH)
dev = 0 if torch.cuda.is_available() else "cpu"
files = sorted(glob.glob(os.path.join(PCD, "*.npz")))
pairs = [(os.path.basename(f), srcpath(os.path.basename(f)[:-4])) for f in files]
pairs = [(k, p) for k, p in pairs if p]
print(f"{len(pairs)}/{len(files)} clouds have a source image", flush=True)

out, B = {}, 64
for i in range(0, len(pairs), B):
    chunk = pairs[i:i + B]
    res = m.predict([p for _, p in chunk], imgsz=640, conf=0.25, device=dev, verbose=False)
    for (key, _), r in zip(chunk, res):
        H, W = r.orig_shape
        if r.obb is not None and len(r.obb):
            j = int(np.argmax(r.obb.conf.cpu().numpy())); x = r.obb.xywhr[j].cpu().numpy()
            out[key] = {"cxf": float(x[0] / W), "cyf": float(x[1] / H),
                        "wf": float(x[2] / W), "hf": float(x[3] / H),
                        "angle": float(x[4]), "conf": float(r.obb.conf[j])}
    if (i // B) % 10 == 0:
        json.dump(out, open(OUTJ, "w"))
        print(f"  {i+len(chunk)}/{len(pairs)} done, {len(out)} with knob", flush=True)
json.dump(out, open(OUTJ, "w"))
print(f"DONE: {len(out)}/{len(pairs)} clouds have a knob -> {OUTJ}", flush=True)
