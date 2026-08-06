#!/usr/bin/env python3
"""Derive a smaller-radius ball dataset from the repaired r08 prepared set.

Unlike re-running ``msecnet_ball/prepare_ball_dataset.py`` from the original
manual JSON, this command treats the current prepared labels and anchors as
the source of truth.  That preserves Web repairs while taking the new-radius
points from each original full cloud.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


ACTIVE_SOURCE_DATASET = "fuelcap_pass_20260803_10847"
REQUIRED_CLOUD_KEYS = {"xyz", "K_norm", "w", "h"}


def ball_mask(xyz, center, radius):
    return np.einsum("ij,ij->i", xyz - center, xyz - center) <= radius ** 2


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare a repaired-label center-ball dataset at a new radius")
    parser.add_argument("prepared_source", type=Path,
                        help="existing prepared r08 dataset whose labels/anchors/split must be preserved")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--source-root", type=Path, default=None,
                        help="original full-cloud root; defaults to dataset.json source_root")
    parser.add_argument("--ball-radius-m", type=float, default=0.05)
    parser.add_argument("--min-ball-points", type=int, default=80)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=None,
                        help="pre-sampling seed; defaults to the source dataset seed")
    args = parser.parse_args(argv)
    source_dir = args.prepared_source.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    if args.ball_radius_m <= 0 or args.min_ball_points < 1 or args.max_points < 0:
        raise ValueError("radius and minimum points must be positive; max points must be >= 0")
    required = ("labels_manual3d.npz", "anchors_manual3d.json", "manifest.jsonl", "clouds",
                "split_by_generalization_group.json", "dataset.json")
    missing = [name for name in required if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"prepared source is missing {missing}")

    metadata = json.loads((source_dir / "dataset.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != "center_ball" or metadata.get("source_dataset") != ACTIVE_SOURCE_DATASET:
        raise ValueError("prepared_source is not the active center-ball dataset")
    source_root = (args.source_root or Path(metadata["source_root"])).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    seed = int(metadata.get("seed", 20260717) if args.seed is None else args.seed)

    with np.load(source_dir / "labels_manual3d.npz", allow_pickle=False) as archive:
        labels = {name: archive[name].copy() for name in archive.files}
    labels_files = [str(value) for value in labels["files"]]
    anchors = json.loads((source_dir / "anchors_manual3d.json").read_text(encoding="utf-8"))
    manifest = [json.loads(line) for line in (source_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if [str(row["file"]) for row in manifest] != labels_files:
        raise ValueError("source manifest and labels order differ")
    split = json.loads((source_dir / "split_by_generalization_group.json").read_text(encoding="utf-8"))
    source_split = {name: set(split.get(name, [])) for name in ("train", "val", "test")}
    if set().union(*source_split.values()) != set(labels_files):
        raise ValueError("source split does not cover exactly the source labels")

    out_clouds = out_dir / "clouds"
    out_clouds.mkdir(parents=True, exist_ok=False)
    new_manifest = []
    cap_counts = np.empty(len(labels_files), dtype=np.int32)
    for index, row in enumerate(manifest):
        file_name = str(row["file"])
        anchor = anchors.get(file_name)
        if not isinstance(anchor, dict) or anchor.get("selection") != "manual_center_ball_patch":
            raise ValueError(f"{file_name} is not a manual center-ball anchor")
        old_radius = float(anchor.get("ball_radius_m", 0))
        if not np.isclose(old_radius, float(metadata["ball_radius_m"]), atol=1e-8, rtol=0):
            raise ValueError(f"{file_name} has inconsistent source anchor radius {old_radius}")
        center = np.asarray(anchor["center_3d"], dtype=np.float32)
        target_normal = np.asarray(anchor["normal"], dtype=np.float32)
        label_index = labels_files.index(file_name)
        label_normal = labels["normal"][label_index].astype(np.float32)
        if abs(float(np.dot(target_normal / np.linalg.norm(target_normal), label_normal / np.linalg.norm(label_normal)))) < 1 - 1e-5:
            raise ValueError(f"{file_name} labels and repaired anchor normal disagree")

        source_cloud = source_root / str(row["source_cloud"])
        with np.load(source_cloud, allow_pickle=False) as cloud:
            if not REQUIRED_CLOUD_KEYS.issubset(cloud.files):
                raise ValueError(f"{source_cloud} lacks {sorted(REQUIRED_CLOUD_KEYS - set(cloud.files))}")
            xyz = cloud["xyz"].astype(np.float32)
            selected = xyz[ball_mask(xyz, center, args.ball_radius_m)]
            source_count = int(len(selected))
            if source_count < args.min_ball_points:
                raise ValueError(f"{file_name} has only {source_count} points in the 5cm ball")
            if args.max_points > 0 and source_count > args.max_points:
                selected = selected[np.random.default_rng(seed + index).choice(source_count, args.max_points, replace=False)]
            np.savez_compressed(
                out_clouds / file_name, xyz=selected, label=np.ones(len(selected), dtype=np.uint8),
                K_norm=cloud["K_norm"], w=cloud["w"], h=cloud["h"],
            )
        cap_counts[index] = len(selected)
        updated_anchor = copy.deepcopy(anchor)
        updated_anchor["ball_radius_m"] = float(args.ball_radius_m)
        updated_anchor["pre_sample_max_points"] = int(args.max_points)
        # Keep the repaired normal/tangent/pose exactly; only the geometric crop radius changes.
        anchors[file_name] = updated_anchor
        updated_row = dict(row)
        updated_row.update({
            "cap_points": int(len(selected)), "source_cap_points": source_count,
            "ball_radius_m": float(args.ball_radius_m), "pre_sample_max_points": int(args.max_points),
        })
        new_manifest.append(updated_row)
        if (index + 1) % 500 == 0 or index + 1 == len(manifest):
            print(f"  clouds {index + 1}/{len(manifest)}", flush=True)

    if "n_inner" in labels:
        labels["n_inner"] = cap_counts
    np.savez_compressed(out_dir / "labels_manual3d.npz", **labels)
    write_json(out_dir / "anchors_manual3d.json", anchors)
    (out_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in new_manifest), encoding="utf-8"
    )
    write_json(out_dir / "split_by_generalization_group.json", split)
    write_json(out_dir / "dataset.json", {
        "schema": "msecnet_prepared_dataset_v1",
        "kind": "center_ball",
        "source_dataset": ACTIVE_SOURCE_DATASET,
        "source_root": str(source_root),
        "source_samples": metadata.get("source_samples"),
        "samples": len(labels_files),
        "ball_radius_m": float(args.ball_radius_m),
        "min_ball_points": int(args.min_ball_points),
        "pre_sample_max_points": int(args.max_points),
        "seed": seed,
        "derived_from": str(source_dir),
        "label_source": "current_repaired_r08_labels_and_anchors",
        "split_source": str(source_dir / "split_by_generalization_group.json"),
    })
    print(f"prepared {len(labels_files)} samples at radius={args.ball_radius_m}m -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
