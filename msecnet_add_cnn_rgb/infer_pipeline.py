#!/usr/bin/env python3
"""YOLO-seg + MSECNet-CNN-RGB inference pipeline.

This is the deployment-style entry point:
RGB image + point-cloud npz -> YOLO inner_cover mask -> normal.
Optionally enable YOLO-OBB knob detection for local patch cropping.
It does not use MoGe/DINO feature files.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from infer_e2e import (  # noqa: E402
    read_rgb,
    predict,
)
from train_cnn_rgb import (  # noqa: E402
    MSECNetWithCNNRGB,
    image_to_tensor,
    project_points_to_image_norm,
)

MROOT = os.path.join(HERE, "..", "msecnet", "MSECNet")
sys.path.insert(0, os.path.join(MROOT, "model"))
sys.path.insert(0, os.path.join(MROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "shared"))

from util import config  # noqa: E402
import cap_patch  # noqa: E402


DEFAULT_SEG_MODEL = os.path.join(ROOT, "models", "seg_4_classes_all_0709_train_aug3", "best.pt")
DEFAULT_OBB_MODEL = os.path.join(ROOT, "models", "inner_obb_clean_v11m_0129", "best.pt")


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
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
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, image_size, ckpt


def run_yolo(image_path, seg_model_path, obb_model_path, device, conf=0.25, imgsz=640, use_obb=False):
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("infer_pipeline.py requires cv2 and ultralytics") from exc

    if not os.path.exists(seg_model_path):
        raise FileNotFoundError(f"missing SEG model: {seg_model_path}")
    if use_obb and not os.path.exists(obb_model_path):
        raise FileNotFoundError(f"missing OBB model: {obb_model_path}")

    yolo_dev = 0 if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    seg = YOLO(seg_model_path)

    r = seg.predict(image_path, imgsz=imgsz, conf=conf, device=yolo_dev, verbose=False)[0]
    h, w = r.orig_shape
    mask = np.zeros((h, w), np.uint8)
    best = -1.0
    if r.masks is not None:
        for i in range(len(r.masks)):
            cls_id = int(r.boxes.cls[i])
            score = float(r.boxes.conf[i])
            if cls_id == 1 and score > best:
                best = score
                m = r.masks.data[i].cpu().numpy()
                mask = (cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)

    kc = None
    knob = False
    if use_obb:
        obb = YOLO(obb_model_path)
        ro = obb.predict(image_path, imgsz=imgsz, conf=conf, device=yolo_dev, verbose=False)[0]
        knob = bool(ro.obb is not None and len(ro.obb) > 0)
        if knob:
            j = int(np.argmax(ro.obb.conf.cpu().numpy()))
            x = ro.obb.xywhr[j].cpu().numpy()
            kc = {
                "cxf": float(x[0] / w),
                "cyf": float(x[1] / h),
                "wf": float(x[2] / w),
                "hf": float(x[3] / h),
                "angle": float(x[4]),
                "conf": float(ro.obb.conf[j]),
            }
    return {
        "mask": mask.astype(bool),
        "inner_conf": float(best),
        "knob": bool(knob),
        "kc": kc,
        "use_obb": bool(use_obb),
        "w": int(w),
        "h": int(h),
    }


def project_xyz_pixels(xyz, K_norm, w, h):
    z = xyz[:, 2]
    u = K_norm[0, 0] * float(w) * xyz[:, 0] / (z + 1e-9) + K_norm[0, 2] * float(w)
    v = K_norm[1, 1] * float(h) * xyz[:, 1] / (z + 1e-9) + K_norm[1, 2] * float(h)
    valid = (z > 1e-6) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    return u, v, valid


def select_points(cloud, yolo, use_yolo_mask=True, use_npz_label=True, radius=0.3, min_points=80):
    xyz_all = cloud["xyz"].astype(np.float32)
    K = cloud["K_norm"]
    w, h = int(cloud["w"]), int(cloud["h"])

    base = np.ones(len(xyz_all), dtype=bool)
    if use_npz_label and "label" in cloud:
        base &= cloud["label"].astype(bool)

    if use_yolo_mask and yolo["inner_conf"] >= 0:
        mask = yolo["mask"]
        if mask.shape != (h, w):
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("cv2 is required to resize YOLO masks") from exc
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        u, v, valid = project_xyz_pixels(xyz_all, K, w, h)
        ui = np.clip(np.round(u).astype(np.int32), 0, w - 1)
        vi = np.clip(np.round(v).astype(np.int32), 0, h - 1)
        base &= valid & mask[vi, ui]

    xyz = xyz_all[base]
    source = "label+yolo_mask" if use_npz_label and use_yolo_mask else "label" if use_npz_label else "all"
    if len(xyz) < min_points:
        if "label" in cloud:
            xyz = xyz_all[cloud["label"].astype(bool)]
            source = "fallback_label"
        if len(xyz) < min_points:
            xyz = xyz_all
            source = "fallback_all"

    kc = yolo.get("kc")
    patch_source = source
    if kc is not None and len(xyz) >= 120:
        _, pm = cap_patch.extract(xyz, K, w, h, kc, radius_frac=radius, min_pts=min_points)
        if pm.sum() >= min_points:
            xyz = xyz[pm]
            patch_source = source + "+knob_patch"
    return xyz.astype(np.float32), patch_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("rgb_image")
    ap.add_argument("cloud_npz")
    ap.add_argument("output_json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seg-model", default=os.environ.get("SEG_MODEL", DEFAULT_SEG_MODEL))
    ap.add_argument("--obb-model", default=os.environ.get("OBB_MODEL", DEFAULT_OBB_MODEL))
    ap.add_argument("--use-obb", action="store_true",
                    help="启用第二个YOLO OBB模型检测knob center，并裁局部patch；默认只用分割mask")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--min-points", type=int, default=80)
    ap.add_argument("--no-yolo-mask", action="store_true",
                    help="不用YOLO inner_cover mask过滤点云；通常不建议")
    ap.add_argument("--no-npz-label", action="store_true",
                    help="不使用cloud npz中的label字段")
    args = ap.parse_args()

    model, image_size, ckpt = load_model(args.checkpoint, args.device)
    yolo = run_yolo(
        args.rgb_image, args.seg_model, args.obb_model, args.device,
        args.conf, args.imgsz, use_obb=args.use_obb
    )
    cloud = np.load(args.cloud_npz)
    xyz, point_source = select_points(
        cloud,
        yolo,
        use_yolo_mask=not args.no_yolo_mask,
        use_npz_label=not args.no_npz_label,
        radius=args.radius,
        min_points=args.min_points,
    )
    if len(xyz) < args.min_points:
        raise RuntimeError(f"too few points after selection: {len(xyz)}")

    rgb = read_rgb(args.rgb_image)
    image_tensor = image_to_tensor(rgb, image_size)
    grid = project_points_to_image_norm(
        xyz, cloud["K_norm"], int(cloud["w"]), int(cloud["h"]), image_size
    )
    normal, centroid = predict(model, xyz, image_tensor, grid, args.device)

    result = {
        "normal": [float(v) for v in normal],
        "centroid": [float(v) for v in centroid],
        "points": int(len(xyz)),
        "point_source": point_source,
        "yolo": {
            "inner_conf": yolo["inner_conf"],
            "knob": yolo["knob"],
            "kc": yolo["kc"],
            "use_obb": yolo["use_obb"],
            "image_w": yolo["w"],
            "image_h": yolo["h"],
        },
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(ckpt.get("step", -1)),
        "rgb_image": args.rgb_image,
        "cloud_npz": args.cloud_npz,
        "image_size": image_size,
        "rgb_backbone": str(ckpt.get("rgb_backbone", "light")),
        "resnet_stage": str(ckpt.get("resnet_stage", "layer2")),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(
        f"normal={np.round(normal, 4).tolist()} points={len(xyz)} "
        f"source={point_source} inner_conf={yolo['inner_conf']:.2f} "
        f"use_obb={yolo['use_obb']} knob={yolo['knob']} -> {args.output_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
