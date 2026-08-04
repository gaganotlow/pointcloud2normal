#!/usr/bin/env python3
"""Export deterministic random center-ball training samples as colored PLYs."""
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
        "comment gray=source points blue=8cm ball training points yellow=human normal",
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


def export_row(dataset_dir, source_root, row_index, row, anchors, raw_points, output_path):
    """Write one source cloud with its prepared ball and target normal highlighted."""
    source_path = source_root / row["source_cloud"]
    training_path = dataset_dir / "clouds" / row["file"]
    if not source_path.is_file() or not training_path.is_file():
        raise FileNotFoundError(source_path if not source_path.is_file() else training_path)

    source_xyz = np.load(source_path)["xyz"].astype(np.float32)
    training_xyz = np.load(training_path)["xyz"].astype(np.float32)
    selected = point_membership(source_xyz, training_xyz)
    if selected.sum() != len(training_xyz):
        raise RuntimeError(f"{row['file']}: saved training points are not an exact subset of the source cloud")

    if raw_points > 0 and len(source_xyz) > raw_points:
        keep = np.random.default_rng(row_index).choice(len(source_xyz), raw_points, replace=False)
        keep = np.unique(np.concatenate((keep, np.flatnonzero(selected))))
        source_xyz, selected = source_xyz[keep], selected[keep]

    rgb = np.tile(GRAY, (len(source_xyz), 1))
    rgb[selected] = BLUE
    anchor = anchors[row["file"]]
    center = np.asarray(anchor["center_3d"], dtype=np.float32)
    normal = np.asarray(anchor["normal"], dtype=np.float32)
    normal /= np.linalg.norm(normal) + 1e-9
    arrow_length = max(0.05, float(anchor["ball_radius_m"]) * 1.25)
    arrow = np.linspace(center, center + normal * arrow_length, 100, dtype=np.float32)
    xyz = np.concatenate((source_xyz, arrow), axis=0)
    rgb = np.concatenate((rgb, np.tile(YELLOW, (len(arrow), 1))), axis=0)
    write_ply(output_path, xyz, rgb)
    return {
        "index": row_index,
        "file": row["file"],
        "split": row["split"],
        "ball_radius_m": float(anchor["ball_radius_m"]),
        "source_points_in_ply": int(len(source_xyz)),
        "ball_points": int(selected.sum()),
        "normal_points": int(len(arrow)),
        "ply": output_path.name,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize random 8 cm center-ball training patches")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--file", default=None, help="export one named prepared cloud instead of random samples")
    parser.add_argument("--random-count", type=int, default=10,
                        help="number of deterministic random samples when --file is absent")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None,
                        help="optionally sample only one data split")
    parser.add_argument("--raw-points", type=int, default=50000,
                        help="maximum gray source points per PLY; all blue ball points are always retained; 0 keeps all")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    anchors = json.loads((dataset_dir / "anchors_manual3d.json").read_text(encoding="utf-8"))
    if args.file is not None and args.random_count != 10:
        parser.error("--file cannot be combined with --random-count")
    if args.file is None and args.random_count < 1:
        parser.error("--random-count must be positive")

    if args.file is not None:
        selected_rows = [next((i, item) for i, item in enumerate(rows) if item["file"] == args.file)]
    else:
        candidates = [(i, row) for i, row in enumerate(rows) if args.split is None or row["split"] == args.split]
        if args.random_count > len(candidates):
            parser.error(f"requested {args.random_count} samples but only {len(candidates)} are available")
        selected_indices = np.random.default_rng(args.seed).choice(len(candidates), args.random_count, replace=False)
        selected_rows = [candidates[int(index)] for index in selected_indices]

    source_root = args.source_root or (dataset_dir.parent / "fuelcap_pass_20260803_10847")
    out_dir = args.out_dir or (dataset_dir / "visualizations" / f"random_ball_seed_{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for order, (row_index, row) in enumerate(selected_rows, start=1):
        out = out_dir / f"sample_{order:02d}_{row_index:05d}_ball.ply"
        item = export_row(dataset_dir, source_root, row_index, row, anchors, args.raw_points, out)
        index.append(item)
        print(f"{order:02d}/{len(selected_rows):02d} file={item['file']} ball_points={item['ball_points']} saved={out}")
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"index={index_path}")


if __name__ == "__main__":
    main()
