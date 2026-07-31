#!/usr/bin/env python3
"""Build an unlabeled MSECNet test set from aligned RGB images and point clouds."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PLY_DTYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
}


def read_xyz_ply(path):
    """Read XYZ from the binary little-endian PLY layout used by this capture set."""
    with open(path, "rb") as f:
        first = f.readline().decode("ascii").strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file")
        vertex_count = None
        vertex_properties = []
        in_vertex = False
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            fields = line.split()
            if fields[:2] == ["format", "binary_little_endian"]:
                continue
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                in_vertex = True
            elif fields and fields[0] == "element":
                in_vertex = False
            elif in_vertex and fields[:1] == ["property"]:
                if len(fields) != 3 or fields[1] not in PLY_DTYPES:
                    raise ValueError(f"unsupported vertex property in {path}: {line}")
                vertex_properties.append((fields[2], "<" + PLY_DTYPES[fields[1]]))
        if vertex_count is None or not {"x", "y", "z"}.issubset(dict(vertex_properties)):
            raise ValueError(f"{path} has no XYZ vertex layout")
        points = np.fromfile(f, dtype=np.dtype(vertex_properties), count=vertex_count)
    return np.column_stack((points["x"], points["y"], points["z"])).astype(np.float32)


def best_obb(result, class_id):
    if result.obb is None or len(result.obb) == 0:
        return None
    classes = result.obb.cls.cpu().numpy().astype(np.int32)
    candidates = np.flatnonzero(classes == class_id)
    if not len(candidates):
        return None
    confidence = result.obb.conf.cpu().numpy()
    index = candidates[np.argmax(confidence[candidates])]
    xywhr = result.obb.xywhr[index].cpu().numpy().astype(np.float32)
    return {
        "cx": float(xywhr[0]), "cy": float(xywhr[1]),
        "w": float(xywhr[2]), "h": float(xywhr[3]), "angle_rad": float(xywhr[4]),
        "confidence": float(confidence[index]), "class_id": int(classes[index]),
    }


def crop_rotated_obb(xyz, intrinsics, obb, expand):
    fx, fy, cx, cy = (float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy"))
    z = xyz[:, 2]
    valid = np.isfinite(xyz).all(axis=1) & (z > 1e-6)
    u = np.empty(len(xyz), dtype=np.float32)
    v = np.empty(len(xyz), dtype=np.float32)
    u[valid] = fx * xyz[valid, 0] / z[valid] + cx
    v[valid] = fy * xyz[valid, 1] / z[valid] + cy
    dx, dy = u - obb["cx"], v - obb["cy"]
    cos_a, sin_a = np.cos(obb["angle_rad"]), np.sin(obb["angle_rad"])
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return valid & (np.abs(local_x) <= obb["w"] * expand / 2) & (np.abs(local_y) <= obb["h"] * expand / 2)


def main():
    parser = argparse.ArgumentParser(description="Prepare an OBB-cropped, unlabeled MSECNet test set")
    parser.add_argument("source_dir", type=Path, help="directory containing per-sample open__* directories")
    parser.add_argument("out_dir", type=Path, help="new output dataset directory; must not already exist")
    parser.add_argument("--obb-model", type=Path, required=True)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--expand", type=float, default=1.10,
                        help="multiply OBB width and height by this value")
    parser.add_argument("--min-points", type=int, default=80)
    parser.add_argument("--max-points", type=int, default=0,
                        help="optional deterministic cap per crop; 0 preserves every cropped point")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="Ultralytics device; default is CUDA:0 when available")
    parser.add_argument("--limit", type=int, default=0, help="optional cap for a smoke test; 0 processes all samples")
    args = parser.parse_args()

    if args.expand <= 0 or args.min_points < 1 or args.max_points < 0 or args.batch_size < 1 or args.limit < 0:
        raise ValueError("invalid crop, point-count, batch-size, or limit argument")
    source_dir, out_dir = args.source_dir.resolve(), args.out_dir.resolve()
    if not source_dir.is_dir() or not args.obb_model.is_file():
        raise FileNotFoundError(source_dir if not source_dir.is_dir() else args.obb_model)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; choose a new output directory")

    samples = [path for path in sorted(source_dir.iterdir()) if path.is_dir()
               and (path / "input_color.jpg").is_file()
               and (path / "camera_intrinsics.json").is_file()
               and (path / "scene_pointcloud.ply").is_file()]
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        raise RuntimeError("no complete RGB/intrinsics/point-cloud samples found")

    out_clouds = out_dir / "clouds"
    out_clouds.mkdir(parents=True)
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    model = YOLO(str(args.obb_model))
    results = model.predict(
        [str(sample / "input_color.jpg") for sample in samples],
        imgsz=640, conf=args.conf, device=device, batch=args.batch_size, verbose=False,
    )

    manifest, anchors, names, skipped = [], {}, [], []
    for index, (sample, result) in enumerate(zip(samples, results), start=1):
        image = cv2.imread(str(sample / "input_color.jpg"), cv2.IMREAD_COLOR)
        if image is None:
            skipped.append({"sample": sample.name, "reason": "unreadable_image"})
            continue
        height, width = image.shape[:2]
        obb = best_obb(result, args.class_id)
        if obb is None:
            skipped.append({"sample": sample.name, "reason": "no_target_obb"})
            continue
        intrinsics = json.loads((sample / "camera_intrinsics.json").read_text(encoding="utf-8"))["camera_intrinsics"]
        xyz = read_xyz_ply(sample / "scene_pointcloud.ply")
        mask = crop_rotated_obb(xyz, intrinsics, obb, args.expand)
        cropped = xyz[mask]
        raw_count = int(len(cropped))
        if raw_count < args.min_points:
            skipped.append({"sample": sample.name, "reason": "too_few_points", "points": raw_count, "obb": obb})
            continue
        if args.max_points and raw_count > args.max_points:
            keep = np.random.default_rng(20260730 + index).choice(raw_count, args.max_points, replace=False)
            cropped = cropped[keep]
        file_name = f"{sample.name}.npz"
        k_norm = np.array([
            [intrinsics["fx"] / width, 0.0, intrinsics["cx"] / width],
            [0.0, intrinsics["fy"] / height, intrinsics["cy"] / height],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        np.savez_compressed(
            out_clouds / file_name,
            xyz=cropped.astype(np.float32), label=np.ones(len(cropped), dtype=np.uint8),
            K_norm=k_norm, w=np.int32(width), h=np.int32(height),
        )
        obb_normalized = {
            "cxf": obb["cx"] / width, "cyf": obb["cy"] / height,
            "wf": obb["w"] / width, "hf": obb["h"] / height,
            "angle": obb["angle_rad"], "conf": obb["confidence"],
        }
        anchors[file_name] = {
            "selection": "projected_obb_2d_crop",
            "obb": obb_normalized,
            "obb_expand": args.expand,
            "source": "yolo_obb",
        }
        manifest.append({
            "file": file_name, "split": "test", "source_dir": sample.name,
            "source_image": f"{sample.name}/input_color.jpg",
            "source_cloud": f"{sample.name}/scene_pointcloud.ply",
            "crop_points": int(len(cropped)), "raw_crop_points": raw_count,
            "obb": obb_normalized, "obb_expand": args.expand,
        })
        names.append(file_name)
        print(f"{index}/{len(samples)} {sample.name}: {raw_count} points", flush=True)

    if not manifest:
        raise RuntimeError("no sample passed OBB detection and point-count validation")
    (out_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8"
    )
    (out_dir / "anchors_obb.json").write_text(json.dumps(anchors, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "split.json").write_text(json.dumps({"test": names}, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "unlabeled_test.npz", files=np.asarray(names))
    metadata = {
        "kind": "msecnet_unlabeled_projected_obb_test",
        "source_dir": str(source_dir), "obb_model": str(args.obb_model.resolve()),
        "class_id": args.class_id, "confidence_threshold": args.conf, "obb_expand": args.expand,
        "samples": len(manifest), "skipped": skipped,
        "note": "No human normal labels are present; use an unlabeled prediction entrypoint, not infer.py metrics.",
    }
    (out_dir / "dataset.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"prepared test={len(manifest)} skipped={len(skipped)} -> {out_dir}")


if __name__ == "__main__":
    main()
