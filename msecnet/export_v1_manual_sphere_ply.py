#!/usr/bin/env python3
"""Export one prepared v1 sample with its human rectangle and training sphere as a colored PLY."""
import argparse
import json
from pathlib import Path

import numpy as np


def sphere_mask_and_radius(xyz, center, radius_frac, min_points=80):
    cap_radius = np.percentile(np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1), 90) + 1e-9
    distance = np.linalg.norm(xyz - center, axis=1)
    frac = radius_frac
    mask = distance < frac * cap_radius
    while mask.sum() < min_points and frac < 1.25:
        frac += 0.15
        mask = distance < frac * cap_radius
    return mask, frac * cap_radius


def manual_innercap_sphere_mask(xyz, center, normal, plane_tol_m, sphere_radius_m):
    delta = xyz - center
    return (np.linalg.norm(delta, axis=1) <= sphere_radius_m) & (np.abs(delta @ normal) <= plane_tol_m)


def line_points(start, end, count=80):
    return np.linspace(start, end, count, dtype=np.float32)


def append_group(points, colors, xyz, color):
    points.append(np.asarray(xyz, dtype=np.float32))
    colors.append(np.tile(np.asarray(color, dtype=np.uint8), (len(xyz), 1)))


def write_ply(path, xyz, rgb):
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("comment gray=raw blue=manual_innercap_candidate green=training_local_sphere\n")
        f.write("comment red=rectangle yellow=normal purple=sphere_wireframe\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(xyz, rgb):
            f.write(f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} {color[0]} {color[1]} {color[2]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--file", default=None, help="prepared cloud filename; default is the first validation sample")
    parser.add_argument("--raw-points", type=int, default=0, help="background points to keep; 0 keeps the full source cloud")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.dataset_dir.resolve()
    anchors = json.loads((root / "anchors_manual3d.json").read_text(encoding="utf-8"))
    manifest = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.file is None:
        row = next(item for item in manifest if item["split"] == "val")
        name = row["file"]
    else:
        name = args.file
        row = next(item for item in manifest if item["file"] == name)
    anchor = anchors[name]
    center = np.asarray(anchor["center_3d"], dtype=np.float32)
    pose = np.asarray(anchor["pose_T"], dtype=np.float32)
    width, height = np.asarray(anchor["rectangle_wh_m"], dtype=np.float32)
    normal = np.asarray(anchor["normal"], dtype=np.float32)

    with np.load(root.parent / "fuelcap_pass_20260717_5873" / row["source_cloud"]) as d:
        raw_full = d["xyz"].astype(np.float32)
    if anchor.get("selection") == "manual_innercap_sphere":
        raw_candidate = manual_innercap_sphere_mask(
            raw_full, center, normal, float(anchor["plane_tol_m"]), float(anchor["inner_radius_m"])
        )
        candidate = raw_full[raw_candidate]
        with np.load(root / "clouds" / name) as d:
            training_candidate = d["xyz"].astype(np.float32)
        compact_sphere, sphere_radius = sphere_mask_and_radius(training_candidate, center, 0.3)
        if compact_sphere.sum() < 80:
            raw_training = raw_candidate
        else:
            raw_training = raw_candidate & (np.linalg.norm(raw_full - center, axis=1) < sphere_radius)
        sphere = raw_training[raw_candidate]
    else:
        with np.load(root / "clouds" / name) as d:
            candidate = d["xyz"].astype(np.float32)
        sphere, sphere_radius = sphere_mask_and_radius(candidate, center, float(anchor["sphere_radius_frac"]))
        raw_candidate = None
        raw_training = None
    raw = raw_full
    if args.raw_points > 0 and len(raw) > args.raw_points:
        indices = np.random.default_rng(0).choice(len(raw), args.raw_points, replace=False)
        raw = raw[indices]
        if raw_candidate is not None:
            raw_candidate = raw_candidate[indices]
            raw_training = raw_training[indices]

    tangent, binormal = pose[:3, 0], pose[:3, 1]
    corners = np.array([
        center - tangent * width / 2 - binormal * height / 2,
        center + tangent * width / 2 - binormal * height / 2,
        center + tangent * width / 2 + binormal * height / 2,
        center - tangent * width / 2 + binormal * height / 2,
    ], dtype=np.float32)

    points, colors = [], []
    if raw_candidate is None:
        append_group(points, colors, raw, (105, 112, 122))
        append_group(points, colors, candidate[~sphere], (45, 125, 220))
        append_group(points, colors, candidate[sphere], (40, 190, 95))
    else:
        # Color the original cloud in place so candidate and training points do not z-fight with gray duplicates.
        points.append(raw)
        raw_colors = np.tile(np.asarray((105, 112, 122), dtype=np.uint8), (len(raw), 1))
        raw_colors[raw_candidate] = np.asarray((45, 125, 220), dtype=np.uint8)
        raw_colors[raw_training] = np.asarray((40, 190, 95), dtype=np.uint8)
        colors.append(raw_colors)
    for i in range(4):
        append_group(points, colors, line_points(corners[i], corners[(i + 1) % 4]), (235, 70, 65))
    arrow_end = center + normal * max(0.05, max(width, height) * 0.8)
    append_group(points, colors, line_points(center, arrow_end), (245, 200, 45))

    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False, dtype=np.float32)
    for axis_a, axis_b in ((tangent, binormal), (tangent, normal), (binormal, normal)):
        ring = center + sphere_radius * (np.cos(theta)[:, None] * axis_a + np.sin(theta)[:, None] * axis_b)
        append_group(points, colors, ring, (170, 75, 210))

    xyz = np.concatenate(points, axis=0)
    rgb = np.concatenate(colors, axis=0)
    out = args.out or (root / "visualizations" / f"{Path(name).stem}_manual_rect_sphere.ply")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out, xyz, rgb)
    print(f"file={name}")
    print(f"candidate_points={len(candidate)} sphere_points={int(sphere.sum())} sphere_radius_m={sphere_radius:.6f}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
