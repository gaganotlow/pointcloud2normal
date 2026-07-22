#!/usr/bin/env python3
"""Export one NormalNet input cloud: gray=raw cloud, blue=points eligible for training."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "shared"))
import cap_patch  # noqa: E402


def write_ply(path, xyz, rgb):
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("comment gray=raw_cloud blue=normalnet_training_points\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(xyz, rgb):
            f.write(f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} {color[0]} {color[1]} {color[2]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", nargs="?", type=Path, default=ROOT / "shared" / "normal_labels_patch03.npz")
    parser.add_argument("pcd_dir", nargs="?", type=Path, default=ROOT / "data" / "pcd_dataset_roi")
    parser.add_argument("--centers", type=Path, default=ROOT / "shared" / "knob_centers.json")
    parser.add_argument("--file", default=None, help="cloud filename; default is the first labels entry")
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--raw-points", type=int, default=0, help="background points to keep; 0 keeps all")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.radius <= 0 or args.raw_points < 0:
        raise ValueError("--radius must be positive and --raw-points must be >= 0")

    with np.load(args.labels) as labels:
        files = [str(name) for name in labels["files"]]
    name = args.file or files[0]
    if name not in set(files):
        raise ValueError(f"{name} is absent from {args.labels}")
    path = args.pcd_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    centers = json.loads(args.centers.read_text(encoding="utf-8")) if args.centers.is_file() else {}

    with np.load(path) as cloud:
        raw = cloud["xyz"].astype(np.float32)
        train_mask = cloud["label"] == 1
        selected_by = "inner_cover_label"
        kc = centers.get(name)
        if kc is not None and int(train_mask.sum()) >= 120:
            inner_indices = np.flatnonzero(train_mask)
            _, patch_mask = cap_patch.extract(
                raw[train_mask], cloud["K_norm"], int(cloud["w"]), int(cloud["h"]), kc, radius_frac=args.radius
            )
            if int(patch_mask.sum()) >= 80:
                train_mask = np.zeros(len(raw), dtype=bool)
                train_mask[inner_indices[patch_mask]] = True
                selected_by = "local_cap_patch"
        if not train_mask.any():
            train_mask = np.ones(len(raw), dtype=bool)
            selected_by = "full_cloud_fallback"

    if args.raw_points > 0 and len(raw) > args.raw_points:
        keep = np.random.default_rng(0).choice(len(raw), args.raw_points, replace=False)
        raw, train_mask = raw[keep], train_mask[keep]
    colors = np.tile(np.asarray((105, 112, 122), dtype=np.uint8), (len(raw), 1))
    colors[train_mask] = np.asarray((45, 125, 220), dtype=np.uint8)
    out = args.out or (HERE / "visualizations" / f"{Path(name).stem}_normalnet_training_points.ply")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out, raw, colors)
    print(f"file={name}")
    print(f"raw_points={len(raw)} training_points={int(train_mask.sum())} selection={selected_by} radius={args.radius:.3f}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
