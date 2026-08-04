#!/usr/bin/env python3
"""Export one pseudo-OBB training sample as a colored PLY."""
import argparse
import json
from pathlib import Path

import numpy as np


GRAY = np.array((105, 112, 122), dtype=np.uint8)
BLUE = np.array((45, 125, 220), dtype=np.uint8)
YELLOW = np.array((245, 200, 45), dtype=np.uint8)


def point_membership(points, subset):
    """Return whether each float32 source point occurs in the saved training patch."""
    point_dtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])
    source_keys = np.ascontiguousarray(points, dtype=np.float32).view(point_dtype).reshape(-1)
    subset_keys = np.ascontiguousarray(subset, dtype=np.float32).view(point_dtype).reshape(-1)
    return np.isin(source_keys, subset_keys)


def write_ply(path, xyz, rgb):
    header = "\n".join((
        "ply",
        "format binary_little_endian 1.0",
        "comment gray=source points blue=pseudo-OBB training points yellow=human normal",
        f"element vertex {len(xyz)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "element face 0",
        "property list uchar int vertex_indices",
        "end_header",
        "",
    )).encode("ascii")
    vertices = np.empty(len(xyz), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    with open(path, "wb") as f:
        f.write(header)
        f.write(vertices.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Visualize a pseudo-OBB training patch")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--file", default=None, help="prepared cloud name; defaults to the first validation sample")
    parser.add_argument("--raw-points", type=int, default=0,
                        help="optional source-cloud cap for a smaller PLY; 0 keeps every source point")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    anchors = json.loads((dataset_dir / "anchors_manual3d.json").read_text(encoding="utf-8"))
    if args.file is None:
        row_index, row = next((i, item) for i, item in enumerate(rows) if item["split"] == "val")
    else:
        row_index, row = next((i, item) for i, item in enumerate(rows) if item["file"] == args.file)

    source_root = args.source_root or (dataset_dir.parent / "fuelcap_pass_20260803_10847")
    source_path = source_root / row["source_cloud"]
    training_path = dataset_dir / "clouds" / row["file"]
    if not source_path.is_file() or not training_path.is_file():
        raise FileNotFoundError(source_path if not source_path.is_file() else training_path)

    source_xyz = np.load(source_path)["xyz"].astype(np.float32)
    training_xyz = np.load(training_path)["xyz"].astype(np.float32)
    selected = point_membership(source_xyz, training_xyz)
    if selected.sum() != len(training_xyz):
        raise RuntimeError("saved training points are not an exact subset of the source cloud")

    if args.raw_points > 0 and len(source_xyz) > args.raw_points:
        keep = np.random.default_rng(0).choice(len(source_xyz), args.raw_points, replace=False)
        keep = np.unique(np.concatenate((keep, np.flatnonzero(selected))))
        source_xyz, selected = source_xyz[keep], selected[keep]

    rgb = np.tile(GRAY, (len(source_xyz), 1))
    rgb[selected] = BLUE
    anchor = anchors[row["file"]]
    center = np.asarray(anchor["center_3d"], dtype=np.float32)
    normal = np.asarray(anchor["normal"], dtype=np.float32)
    normal /= np.linalg.norm(normal) + 1e-9
    arrow_length = max(0.05, float(max(anchor["rectangle_wh_m"])) * 1.2)
    arrow = np.linspace(center, center + normal * arrow_length, 100, dtype=np.float32)
    source_xyz = np.concatenate((source_xyz, arrow), axis=0)
    rgb = np.concatenate((rgb, np.tile(YELLOW, (len(arrow), 1))), axis=0)
    out = args.out or (dataset_dir / "visualizations" / f"sample_{row_index:05d}_pseudo_obb.ply")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out, source_xyz, rgb)
    print(f"file={row['file']}")
    print(f"source_points={len(source_xyz) - len(arrow)} training_points={int(selected.sum())} normal_points={len(arrow)}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
