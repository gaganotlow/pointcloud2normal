#!/usr/bin/env python3
"""Prepare the reviewed fuel-cap export for train_v1 with human 3D rectangle anchors."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REQUIRED_CLOUD_KEYS = {"xyz", "label", "K_norm", "w", "h"}


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def assign_model_splits(rows, seed):
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["car_model"]].append(row)
    rng = np.random.default_rng(seed)
    models = list(by_model)
    rng.shuffle(models)
    tie_breaker = {model: i for i, model in enumerate(models)}
    sizes = {model: len(items) for model, items in by_model.items()}

    def choose_eval_models(candidates, target_samples, min_models=8):
        chosen = []
        total = 0
        while candidates and (total < target_samples or len(chosen) < min_models):
            remaining = target_samples - total
            under_target = [model for model in candidates if sizes[model] <= remaining]
            options = under_target or candidates
            model = min(
                options,
                key=lambda name: (abs(total + sizes[name] - target_samples), tie_breaker[name]),
            )
            candidates.remove(model)
            chosen.append(model)
            total += sizes[model]
        return chosen

    candidates = set(models)
    target_samples = max(1, round(len(rows) * 0.1))
    val_models = choose_eval_models(candidates, target_samples)
    test_models = choose_eval_models(candidates, target_samples)
    return {
        model: "val" if model in val_models else "test" if model in test_models else "train"
        for model in models
    }


def manual_innercap_sphere_mask(xyz, center_3d, normal, plane_tol_m, sphere_radius_m):
    """Select the manually anchored, 3D spherical inner-cap region from the full cloud."""
    delta = xyz - center_3d
    return (np.linalg.norm(delta, axis=1) <= sphere_radius_m) & (np.abs(delta @ normal) <= plane_tol_m)


def load_and_validate(source_root, row, max_center_z, min_cap_points, plane_tol_m, sphere_radius_m):
    cloud_path = source_root / row["cloud_full"]
    label_path = source_root / row["label"]
    if not cloud_path.is_file() or not label_path.is_file():
        return None, "missing_source"

    label = json.loads(label_path.read_text(encoding="utf-8"))
    required = ("normal", "center", "tangent", "pose_T", "disc_wh")
    if any(key not in label for key in required):
        return None, "missing_manual_rectangle"
    normal = np.asarray(label["normal"], dtype=np.float64)
    center = np.asarray(label["center"], dtype=np.float64)
    tangent = np.asarray(label["tangent"], dtype=np.float64)
    pose = np.asarray(label["pose_T"], dtype=np.float64)
    wh = np.asarray(label["disc_wh"], dtype=np.float64)
    if normal.shape != (3,) or center.shape != (3,) or tangent.shape != (3,) or pose.shape != (4, 4) or wh.shape != (2,):
        return None, "invalid_manual_rectangle_shape"
    if not all(np.isfinite(v).all() for v in (normal, center, tangent, pose, wh)) or np.any(wh <= 0):
        return None, "invalid_manual_rectangle_values"
    if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-3) or normal @ (-center) <= 0:
        return None, "invalid_manual_normal"
    if np.max(np.abs(pose[:3, 3] - center)) > 2e-3 or pose[:3, 2] @ normal < 0.999:
        return None, "manual_pose_mismatch"
    if max_center_z > 0 and center[2] > max_center_z:
        return None, "center_z_too_large"

    try:
        with np.load(cloud_path) as cloud:
            if not REQUIRED_CLOUD_KEYS.issubset(cloud.files):
                return None, "missing_cloud_fields"
            xyz = cloud["xyz"]
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                return None, "invalid_cloud_shape"
            cap_points = int(np.count_nonzero(manual_innercap_sphere_mask(
                xyz, center.astype(np.float32), normal.astype(np.float32), plane_tol_m, sphere_radius_m
            )))
    except Exception:
        return None, "unreadable_cloud"
    if cap_points < min_cap_points:
        return None, "too_few_cap_points"

    return {
        "file": f"{row['id']}.npz",
        "source_cloud": str(row["cloud_full"]),
        "source_label": str(row["label"]),
        "dataset": row["dataset"],
        "car_model": row["car_model"],
        "normal": normal.astype(np.float32),
        "center": center.astype(np.float32),
        "tangent": tangent.astype(np.float32),
        "pose": pose.astype(np.float32),
        "disc_wh": wh.astype(np.float32),
        "cap_points": cap_points,
        "cloud_path": cloud_path,
    }, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max-center-z", type=float, default=3.0,
                        help="exclude known scale-corrupted samples; <=0 keeps all depths")
    parser.add_argument("--min-cap-points", type=int, default=80)
    parser.add_argument("--plane-tol-m", type=float, default=0.005,
                        help="half-thickness around the human annotated plane")
    parser.add_argument("--inner-radius-m", type=float, default=0.10,
                        help="radius of the human-centered inner-cap sphere in meters")
    parser.add_argument("--max-points", type=int, default=4096,
                        help="maximum pre-sampled points per inner-cap cloud; 0 keeps all")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    index_path = source_root / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if args.out_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.out_dir}")
    if args.plane_tol_m <= 0 or args.inner_radius_m <= 0 or args.max_points < 0:
        raise ValueError("--plane-tol-m and --inner-radius-m must be positive; --max-points must be >= 0")

    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line]
    selected, skipped = [], defaultdict(int)
    for row in rows:
        item, reason = load_and_validate(
            source_root, row, args.max_center_z, args.min_cap_points,
            args.plane_tol_m, args.inner_radius_m,
        )
        if item is None:
            skipped[reason] += 1
        else:
            selected.append(item)
    if not selected:
        raise RuntimeError("no usable samples after validation")
    if len({item["file"] for item in selected}) != len(selected):
        raise RuntimeError("duplicate destination cloud filenames")

    model_split = assign_model_splits(selected, args.seed)
    split = {"train": [], "val": [], "test": []}
    for item in selected:
        split[model_split[item["car_model"]]].append(item["file"])
    if not split["train"] or not split["val"]:
        raise RuntimeError("car-model split did not produce train and val samples")

    out_dir = args.out_dir.resolve()
    cloud_dir = out_dir / "clouds"
    cloud_dir.mkdir(parents=True)
    anchors = {}
    manifest = []
    for item_index, item in enumerate(selected):
        destination = cloud_dir / item["file"]
        with np.load(item["cloud_path"]) as cloud:
            xyz = cloud["xyz"].astype(np.float32)
            mask = manual_innercap_sphere_mask(
                xyz, item["center"], item["normal"], args.plane_tol_m, args.inner_radius_m
            )
            xyz = xyz[mask]
            item["source_cap_points"] = int(len(xyz))
            if args.max_points > 0 and len(xyz) > args.max_points:
                xyz = xyz[np.random.default_rng(args.seed + item_index).choice(len(xyz), args.max_points, replace=False)]
            item["cap_points"] = int(len(xyz))
            np.savez_compressed(
                destination,
                xyz=xyz,
                label=np.ones(len(xyz), dtype=np.uint8),
                K_norm=cloud["K_norm"],
                w=cloud["w"],
                h=cloud["h"],
            )
        anchors[item["file"]] = {
            "center_3d": item["center"].tolist(),
            "rectangle_wh_m": item["disc_wh"].tolist(),
            "tangent": item["tangent"].tolist(),
            "normal": item["normal"].tolist(),
            "pose_T": item["pose"].tolist(),
            "source": "human_3d_rectangle",
            "selection": "manual_innercap_sphere",
            "plane_tol_m": args.plane_tol_m,
            "inner_radius_m": args.inner_radius_m,
            "pre_sample_max_points": args.max_points,
        }
        manifest.append({
            "file": item["file"],
            "split": model_split[item["car_model"]],
            "dataset": item["dataset"],
            "car_model": item["car_model"],
            "source_cloud": item["source_cloud"],
            "source_label": item["source_label"],
            "cap_points": item["cap_points"],
            "source_cap_points": item["source_cap_points"],
            "label_source": "human_3d_innercap_sphere",
            "plane_tol_m": args.plane_tol_m,
            "inner_radius_m": args.inner_radius_m,
            "pre_sample_max_points": args.max_points,
        })

    names = np.array([item["file"] for item in selected])
    normals = np.stack([item["normal"] for item in selected])
    cap_counts = np.array([item["cap_points"] for item in selected], dtype=np.int32)
    np.savez_compressed(
        out_dir / "labels_manual3d.npz",
        files=names,
        normal=normals,
        inlier_frac=np.ones(len(selected), dtype=np.float32),
        agree_deg=np.zeros(len(selected), dtype=np.float32),
        n_inner=cap_counts,
    )
    write_json(out_dir / "anchors_manual3d.json", anchors)
    split["seed"] = args.seed
    split["model_split"] = model_split
    write_json(out_dir / "split_by_car_model.json", split)
    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {name: len(split[name]) for name in ("train", "val", "test")}
    readme = "\n".join([
        "# MSECNet v1 Manual-3D Dataset",
        "",
        "Prepared exclusively from fuelcap_pass_20260717_5873.",
        "",
        "- clouds/: 10 cm inner-cap sphere clouds, uniformly pre-sampled to at most 4096 points.",
        "- labels_manual3d.npz: human normal targets for train_v1.",
        "- anchors_manual3d.json: records the human center, normal, and 10 cm inner-cap sphere.",
        "- split_by_car_model.json: car-model-disjoint train/val/test split.",
        "- manifest.jsonl: source traceability for every retained cloud.",
        "",
        f"Retained samples: {len(selected)}",
        f"Split: train={counts['train']} val={counts['val']} test={counts['test']}",
        f"Excluded center.z > {args.max_center_z} m: {skipped['center_z_too_large']}",
        f"Point mask: distance(center) <= {args.inner_radius_m} m and abs(normal_distance) <= {args.plane_tol_m} m.",
        f"Each cloud is pre-sampled to at most {args.max_points} points. train_v1 then applies its 0.3 * capR sphere around the human center and randomly selects 1024 points.",
        "",
        "Train with:",
        "python msecnet/train_v1.py DATASET/labels_manual3d.npz DATASET/clouds "
        "--centers DATASET/anchors_manual3d.json --split DATASET/split_by_car_model.json",
        "",
    ])
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"prepared {len(selected)}/{len(rows)} samples -> {out_dir}")
    print("split " + " ".join(f"{name}={counts[name]}" for name in ("train", "val", "test")))
    print("skipped " + json.dumps(dict(sorted(skipped.items())), ensure_ascii=False))


if __name__ == "__main__":
    main()
