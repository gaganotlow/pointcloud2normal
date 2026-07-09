#!/usr/bin/env python3
"""YOLO-seg + online MoGe + MSECNet-RGB inference pipeline.

Deployment-style input:
RGB image + point-cloud npz -> YOLO inner_cover mask -> MoGe/DINO features -> normal.

This keeps the training-time feature definition exactly the same as
train_rgb_fusion.py, but can extract the MoGe feature for one image on demand.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MROOT = os.path.join(HERE, "..", "msecnet", "MSECNet")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(MROOT, "model"))
sys.path.insert(0, os.path.join(MROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "shared"))

from util import config  # noqa: E402
import cap_patch  # noqa: E402
from precompute_moge_feat import WEIGHTS, extract_encoder_features  # noqa: E402
from train_rgb_fusion import (  # noqa: E402
    MSECNetWithRGB,
    RGB_MODES,
    project_features_to_points,
    rgb_dim_from_parts,
)


DEFAULT_SEG_MODEL = os.path.join(ROOT, "models", "seg_4_classes_all_0709_train_aug3", "best.pt")
DEFAULT_OBB_MODEL = os.path.join(ROOT, "models", "inner_obb_clean_v11m_0129", "best.pt")


def resolve_device(device):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: {device} requested but CUDA is unavailable; using cpu", flush=True)
        return "cpu"
    return device


def read_rgb(path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 is required to read RGB images") from exc
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def image_to_moge_tensor(rgb):
    rgb = np.asarray(rgb)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1)
    if t.max() > 1.5:
        t = t / 255.0
    return t


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


def empty_yolo_info(image_shape=None):
    h, w = image_shape[:2] if image_shape is not None else (0, 0)
    return {
        "mask": np.zeros((int(h), int(w)), dtype=bool),
        "inner_conf": -1.0,
        "knob": False,
        "kc": None,
        "use_obb": False,
        "w": int(w),
        "h": int(h),
    }


def project_xyz_pixels(xyz, K_norm, w, h):
    xyz = np.asarray(xyz, dtype=np.float32)
    K_norm = np.asarray(K_norm, dtype=np.float32)
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

    yolo_used = bool(use_yolo_mask and yolo.get("inner_conf", -1.0) >= 0)
    if yolo_used:
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

    if use_npz_label and yolo_used:
        source = "label+yolo_mask"
    elif use_npz_label:
        source = "label"
    elif yolo_used:
        source = "yolo_mask"
    else:
        source = "all"

    xyz = xyz_all[base]
    if len(xyz) < min_points:
        if use_npz_label and "label" in cloud:
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


def default_cache_path(cache_dir, cloud_npz):
    return os.path.join(cache_dir, os.path.basename(cloud_npz))


def load_cached_moge(path, image_h=None, image_w=None):
    if not path or not os.path.exists(path):
        return None
    try:
        with np.load(path) as d:
            if "feat_map" not in d or "cls_token" not in d:
                return None
            cached_h = int(d["h"]) if "h" in d else None
            cached_w = int(d["w"]) if "w" in d else None
            if image_h is not None and cached_h is not None and cached_h != int(image_h):
                return None
            if image_w is not None and cached_w is not None and cached_w != int(image_w):
                return None
            feat_map = d["feat_map"].astype(np.float32)
            cls_token = d["cls_token"].astype(np.float32)
            meta = {
                "cache_hit": True,
                "cache_path": path,
                "feat_dim": int(feat_map.shape[0]),
                "cls_dim": int(cls_token.shape[0]),
                "h": cached_h,
                "w": cached_w,
            }
            return feat_map, cls_token, meta
    except Exception as exc:
        print(f"WARNING: failed to read cached MoGe feature {path}: {exc}", flush=True)
        return None


def save_moge_cache(path, feat_map, cls_token, h, w, save_dtype, num_tokens, resolution_level):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if save_dtype == "float16":
        feat_save = feat_map.astype(np.float16)
        cls_save = cls_token.astype(np.float16)
    else:
        feat_save = feat_map.astype(np.float32)
        cls_save = cls_token.astype(np.float32)

    tmp_path = f"{path}.tmp.{os.getpid()}.npz"
    np.savez_compressed(
        tmp_path,
        feat_map=feat_save,
        cls_token=cls_save,
        h=np.array(h, dtype=np.int32),
        w=np.array(w, dtype=np.int32),
        feat_dim=np.array(feat_save.shape[0], dtype=np.int32),
        cls_dim=np.array(cls_save.shape[0], dtype=np.int32),
        weights=np.array(WEIGHTS),
        save_dtype=np.array(save_dtype),
        num_tokens=np.array(-1 if num_tokens is None else num_tokens, dtype=np.int32),
        resolution_level=np.array(resolution_level, dtype=np.int32),
    )
    os.replace(tmp_path, path)


def load_moge_model(device):
    try:
        from moge.model.v2 import MoGeModel
    except ImportError as exc:
        raise RuntimeError("online MoGe extraction requires the moge package") from exc
    print(f"Loading MoGe model: {WEIGHTS}", flush=True)
    return MoGeModel.from_pretrained(WEIGHTS).to(device).eval()


def get_moge_features(rgb, cloud_npz, args):
    h, w = rgb.shape[:2]
    cache_path = default_cache_path(args.moge_cache_dir, cloud_npz) if args.moge_cache_dir else None
    if cache_path and not args.overwrite_moge_cache:
        cached = load_cached_moge(cache_path, image_h=h, image_w=w)
        if cached is not None:
            return cached

    moge = load_moge_model(args.device)
    rgb_tensor = image_to_moge_tensor(rgb)
    feat_map, cls_token = extract_encoder_features(
        moge,
        rgb_tensor,
        args.device,
        num_tokens=args.num_tokens,
        resolution_level=args.resolution_level,
    )
    meta = {
        "cache_hit": False,
        "cache_path": cache_path,
        "feat_dim": int(feat_map.shape[0]),
        "cls_dim": int(cls_token.shape[0]),
        "h": int(h),
        "w": int(w),
        "weights": WEIGHTS,
        "num_tokens": args.num_tokens,
        "resolution_level": int(args.resolution_level),
    }
    if cache_path:
        save_moge_cache(
            cache_path,
            feat_map,
            cls_token,
            h,
            w,
            args.save_cache_dtype,
            args.num_tokens,
            args.resolution_level,
        )
    return feat_map, cls_token, meta


def build_rgb_global_from_moge(xyz, cloud, feat_map, cls_token, rgb_feat_dim, mode):
    if mode not in RGB_MODES:
        raise ValueError(f"mode must be one of {RGB_MODES}, got {mode}")
    if len(xyz) == 0:
        return np.zeros(rgb_feat_dim, dtype=np.float32)

    parts = []
    if mode in ("full", "map"):
        point_feats = project_features_to_points(
            xyz,
            feat_map,
            cloud["K_norm"],
            int(cloud["w"]),
            int(cloud["h"]),
        )
        parts.extend([point_feats.max(axis=0), point_feats.mean(axis=0)])
    if mode in ("full", "cls"):
        parts.append(cls_token.astype(np.float32))

    rgb_global = np.concatenate(parts, axis=0).astype(np.float32)
    rgb_global = np.nan_to_num(rgb_global, copy=False)
    if rgb_global.shape[0] != int(rgb_feat_dim):
        raise ValueError(
            f"MoGe descriptor dim mismatch: got {rgb_global.shape[0]}, expected {rgb_feat_dim}. "
            f"mode={mode} feat_dim={feat_map.shape[0]} cls_dim={cls_token.shape[0]}"
        )
    return rgb_global


def load_model_from_checkpoint(ckpt, rgb_feat_dim, device):
    cfg = config.load_cfg_from_cfg_file(os.path.join(MROOT, "scripts/config/pcpnet/config.yaml"))
    cfg.num_classes = 3
    model = MSECNetWithRGB(cfg, rgb_feat_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, xyz, rgb_global, device):
    centroid = xyz.mean(0).astype(np.float32)
    xyz_norm = xyz - centroid
    xyz_norm = xyz_norm / (np.linalg.norm(xyz_norm, axis=1).max() + 1e-9)

    coord = torch.from_numpy(xyz_norm.astype(np.float32)).float().to(device)
    offset = torch.tensor([len(xyz_norm)], dtype=torch.int32, device=device)
    rgb_global_t = torch.from_numpy(rgb_global).float().unsqueeze(0).to(device)

    pp = model(coord, torch.zeros(coord.shape[0], 0, device=device), offset, rgb_global_t)
    pp = F.normalize(pp, dim=1).cpu().numpy()

    camdir = -centroid / (np.linalg.norm(centroid) + 1e-9)
    flip = (pp @ camdir) < 0
    pp[flip] = -pp[flip]
    normal = pp.mean(0)
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    if normal @ camdir < 0:
        normal = -normal
    return normal.astype(np.float32), centroid


def aspect_warning(rgb_shape, cloud):
    image_h, image_w = rgb_shape[:2]
    cloud_w, cloud_h = int(cloud["w"]), int(cloud["h"])
    image_aspect = float(image_w) / max(float(image_h), 1.0)
    cloud_aspect = float(cloud_w) / max(float(cloud_h), 1.0)
    if abs(image_aspect - cloud_aspect) / max(cloud_aspect, 1e-9) > 0.02:
        return (
            f"RGB aspect {image_w}x{image_h} differs from cloud camera size "
            f"{cloud_w}x{cloud_h}; projection may be misaligned"
        )
    return None


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
                    help="run a second YOLO OBB model for knob-center patch cropping")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--min-points", type=int, default=80)
    ap.add_argument("--no-yolo", action="store_true",
                    help="skip YOLO entirely and use npz label/all points only")
    ap.add_argument("--no-yolo-mask", action="store_true",
                    help="run YOLO for diagnostics but do not use the inner_cover mask")
    ap.add_argument("--no-npz-label", action="store_true",
                    help="do not use the cloud npz label field")
    ap.add_argument("--rgb-mode", choices=RGB_MODES, default=None,
                    help="default reads rgb_mode from checkpoint")
    ap.add_argument("--moge-cache-dir", default=None,
                    help="optional feature cache dir; filename defaults to basename(cloud_npz)")
    ap.add_argument("--overwrite-moge-cache", action="store_true")
    ap.add_argument("--num-tokens", type=int, default=None,
                    help="MoGe/DINO encoder token count; default follows resolution level")
    ap.add_argument("--resolution-level", type=int, default=9,
                    help="used when --num-tokens is not set")
    ap.add_argument("--save-cache-dtype", choices=("float16", "float32"), default="float16")
    args = ap.parse_args()
    args.device = resolve_device(args.device)

    rgb = read_rgb(args.rgb_image)
    cloud = np.load(args.cloud_npz)
    warnings = []
    warn = aspect_warning(rgb.shape, cloud)
    if warn:
        warnings.append(warn)
        print("WARNING: " + warn, flush=True)

    if args.no_yolo:
        yolo = empty_yolo_info(rgb.shape)
    else:
        yolo = run_yolo(
            args.rgb_image,
            args.seg_model,
            args.obb_model,
            args.device,
            conf=args.conf,
            imgsz=args.imgsz,
            use_obb=args.use_obb,
        )

    xyz, point_source = select_points(
        cloud,
        yolo,
        use_yolo_mask=(not args.no_yolo and not args.no_yolo_mask),
        use_npz_label=not args.no_npz_label,
        radius=args.radius,
        min_points=args.min_points,
    )
    if len(xyz) < args.min_points:
        raise RuntimeError(f"too few points after selection: {len(xyz)}")

    feat_map, cls_token, moge_meta = get_moge_features(rgb, args.cloud_npz, args)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    rgb_mode = args.rgb_mode or ckpt.get("rgb_mode", "full")
    if rgb_mode in ("full", "map") and feat_map.shape[0] <= 0:
        raise RuntimeError(f"rgb_mode={rgb_mode} needs a MoGe feature map")
    if rgb_mode in ("full", "cls") and cls_token.shape[0] <= 0:
        raise RuntimeError(f"rgb_mode={rgb_mode} needs a MoGe CLS token")

    inferred_dim = rgb_dim_from_parts(feat_map.shape[0], cls_token.shape[0], rgb_mode)
    rgb_feat_dim = int(ckpt.get("rgb_feat_dim", inferred_dim))
    if rgb_feat_dim != inferred_dim:
        raise ValueError(
            f"checkpoint expects rgb_feat_dim={rgb_feat_dim}, but current MoGe descriptor "
            f"has {inferred_dim} for mode={rgb_mode}"
        )

    rgb_global = build_rgb_global_from_moge(
        xyz,
        cloud,
        feat_map,
        cls_token,
        rgb_feat_dim,
        rgb_mode,
    )
    model = load_model_from_checkpoint(ckpt, rgb_feat_dim, args.device)
    normal, centroid = predict(model, xyz, rgb_global, args.device)

    result = {
        "normal": [float(v) for v in normal],
        "centroid": [float(v) for v in centroid],
        "points": int(len(xyz)),
        "point_source": point_source,
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(ckpt.get("step", -1)),
        "checkpoint_mean_err": float(ckpt["mean_err"]) if "mean_err" in ckpt else None,
        "rgb_image": args.rgb_image,
        "cloud_npz": args.cloud_npz,
        "rgb_mode": rgb_mode,
        "rgb_feat_dim": int(rgb_feat_dim),
        "moge": moge_meta,
        "yolo": {
            "enabled": not args.no_yolo,
            "inner_conf": yolo["inner_conf"],
            "knob": yolo["knob"],
            "kc": yolo["kc"],
            "use_obb": yolo["use_obb"],
            "image_w": yolo["w"],
            "image_h": yolo["h"],
        },
        "warnings": warnings,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"normal={np.round(normal, 4).tolist()} points={len(xyz)} "
        f"source={point_source} rgb_mode={rgb_mode} "
        f"moge_cache_hit={moge_meta.get('cache_hit')} "
        f"inner_conf={yolo['inner_conf']:.2f} -> {args.output_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
