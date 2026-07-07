#!/usr/bin/env python3
"""Stage A (env 01_yolo_seg): new seg(inner_cover) + knob OBB. Usage: _segobb.py <image> <workdir>"""
import json
import os
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SEG_MODEL = os.environ.get("SEG_MODEL",
    os.path.join(ROOT, "models", "seg_4_classes_all_0709_train_aug3", "best.pt"))
OBB_MODEL = os.environ.get("OBB_MODEL",
    os.path.join(ROOT, "models", "inner_obb_clean_v11m_0129", "best.pt"))

img, wd = sys.argv[1], sys.argv[2]
os.makedirs(wd, exist_ok=True)
dev = 0 if torch.cuda.is_available() else "cpu"
seg = YOLO(SEG_MODEL)
obb = YOLO(OBB_MODEL)
r = seg.predict(img, imgsz=640, conf=0.25, device=dev, verbose=False)[0]
h, w = r.orig_shape
mask = np.zeros((h, w), np.uint8); best = -1.0
if r.masks is not None:
    for i in range(len(r.masks)):
        if int(r.boxes.cls[i]) == 1 and float(r.boxes.conf[i]) > best:
            best = float(r.boxes.conf[i])
            mask = (cv2.resize(r.masks.data[i].cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8) * 255
ro = obb.predict(img, imgsz=640, conf=0.25, device=dev, verbose=False)[0]
knob = bool(ro.obb is not None and len(ro.obb) > 0)
kc = None
if knob:
    j = int(np.argmax(ro.obb.conf.cpu().numpy())); x = ro.obb.xywhr[j].cpu().numpy()
    kc = {"cxf": float(x[0] / w), "cyf": float(x[1] / h), "conf": float(ro.obb.conf[j])}
cv2.imwrite(wd + "/mask_inner.png", mask)
json.dump({"knob": knob, "inner_conf": best, "w": int(w), "h": int(h), "kc": kc}, open(wd + "/segobb.json", "w"))
print(f"knob={knob} inner_conf={best:.2f} kc={kc}")
