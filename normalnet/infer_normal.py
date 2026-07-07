#!/usr/bin/env python3
"""Orchestrate the outward-normal inference: seg+OBB -> MoGe -> NormalNet -> arrow viz.
Only runs pose when BOTH inner_cover seg and knob OBB are detected.

All stages run in a single conda environment (point2normal).

Usage:
    python normalnet/infer_normal.py <img...>
    python normalnet/infer_normal.py img1.png img2.png

Env vars:
    NORMALNET_CKPT  -- path to NormalNet checkpoint (default: normalnet/ckpt_normal/best.pt)
    CUDA_VISIBLE_DEVICES
    HF_HOME          -- HuggingFace cache dir
    SEG_MODEL        -- YOLO seg model path
    OBB_MODEL        -- YOLO OBB model path
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PYTHON = sys.executable  # single conda env for all stages

CKPT = os.environ.get("NORMALNET_CKPT", os.path.join(HERE, "ckpt_normal", "best.pt"))

# Default: look for test images; override with CLI args
imgs = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(ROOT, "data", "eval_depth", "*.png")))[:6]
env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
           HF_HOME=os.environ.get("HF_HOME", os.path.join(ROOT, "models", "hf_cache")))

for img in imgs:
    stem = os.path.splitext(os.path.basename(img))[0]
    wd = os.path.join(ROOT, "output", "runs_infer", stem)
    os.makedirs(wd, exist_ok=True)
    print(f"\n==== {stem[:46]} ====", flush=True)
    subprocess.run([PYTHON, os.path.join(HERE, "_segobb.py"), img, wd], env=env)
    meta = json.load(open(os.path.join(wd, "segobb.json")))
    if not meta["knob"] or meta["inner_conf"] < 0:
        print("  no knob / no inner_cover -> skip (no pose)"); continue
    subprocess.run([PYTHON, os.path.join(HERE, "s2_moge.py"), img, wd], env=env,
                   stdout=subprocess.DEVNULL)
    subprocess.run([PYTHON, os.path.join(HERE, "_norminfer.py"), wd, img, CKPT], env=env)
print(f"\nresults -> {os.path.join(ROOT, 'output', 'normal_pred_demo')}/")
