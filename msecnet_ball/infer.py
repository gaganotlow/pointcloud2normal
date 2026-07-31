#!/usr/bin/env python3
"""Evaluate a center-anchored spherical MSECNet on a named dataset split."""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(HERE).parent
MROOT = os.path.join(PROJECT_ROOT, "msecnet_best", "MSECNet")
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "msecnet_ball_v1_fuelcap_pass_20260717_manual3d_r08"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "msecnet_ball" / "out" / "center_ball_r08_oriented_v1" / "best.pt"
DEFAULT_SPLIT = DEFAULT_DATASET_DIR / "split_by_generalization_group.json"
sys.path.insert(0, os.path.join(MROOT, "model"))
sys.path.insert(0, os.path.join(MROOT, "scripts"))

from architectures import MSECNet  # noqa: E402
from util import config  # noqa: E402
from train import (  # noqa: E402
    BallNormalDS,
    aggregate_point_normals,
    collate_variable_points,
    oriented_angular_error_deg,
    radial_weights,
    resolve_max_points,
    to_msecnet,
)


def load_split_indices(files, split_path, split_name):
    """Return label indices for one explicit split and validate its contents."""
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    if split_name not in split:
        raise ValueError(f"{split_path} has no '{split_name}' split")
    selected = set(split[split_name])
    names = [str(name) for name in files]
    known = set(names)
    unknown = selected - known
    if unknown:
        raise ValueError(f"{split_path} references {len(unknown)} files absent from labels")
    indices = np.array([i for i, name in enumerate(names) if name in selected], dtype=np.int64)
    if not len(indices):
        raise ValueError(f"split '{split_name}' is empty")
    if len(indices) != len(selected):
        raise ValueError(f"duplicate file names in labels for split '{split_name}'")
    return indices


def load_models(manifest_path):
    if not manifest_path.is_file():
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["file"]: row.get("car_model", "") for row in rows}


def summarize(rows):
    errors = np.array([row.get("angular_error_deg", row["axis_error_deg"]) for row in rows], dtype=np.float64)
    return {
        "samples": int(len(rows)),
        "mean_angular_error_deg": float(errors.mean()),
        "median_angular_error_deg": float(np.median(errors)),
        "acc10_pct": float((errors <= 10.0).mean() * 100.0),
        # Kept for the existing read-only web report and prior integrations.
        "mean_axis_error_deg": float(errors.mean()),
        "median_axis_error_deg": float(np.median(errors)),
    }


@torch.no_grad()
def infer(model, loader, device, file_names, model_by_file, radial_weight_beta):
    model.eval()
    rows = []
    sample_offset = 0
    for coord, radial_distance, offset, target, _, counts in loader:
        coord, feat, offset = to_msecnet(
            coord.to(device, non_blocking=True), radial_distance.to(device, non_blocking=True), offset.to(device, non_blocking=True)
        )
        point_vectors = model(coord, feat, offset)
        mean_vector, normal, _ = aggregate_point_normals(
            point_vectors, counts, radial_weights(feat, radial_weight_beta)
        )
        target = target.to(device, non_blocking=True)
        errors = oriented_angular_error_deg(normal, target)
        confidence = mean_vector.norm(dim=1)
        for batch_index in range(len(counts)):
            name = str(file_names[sample_offset + batch_index])
            rows.append({
                "file": name,
                "car_model": model_by_file.get(name, ""),
                "pred_normal": [float(v) for v in normal[batch_index].cpu().tolist()],
                "target_normal": [float(v) for v in target[batch_index].cpu().tolist()],
                "angular_error_deg": float(errors[batch_index].cpu()),
                "axis_error_deg": float(errors[batch_index].cpu()),
                "point_consensus": float(confidence[batch_index].cpu()),
                "mean_vector_norm": float(confidence[batch_index].cpu()),
            })
        sample_offset += len(counts)
    if sample_offset != len(file_names):
        raise RuntimeError(f"inference returned {sample_offset} samples; expected {len(file_names)}")
    return rows


def write_outputs(out_dir, metadata, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["car_model"]].append(row)
    report = {
        "metadata": metadata,
        "summary": summarize(rows),
        "by_car_model": {name: summarize(items) for name, items in sorted(by_model.items())},
        "predictions": rows,
    }
    json_path = out_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "car_model", "pred_nx", "pred_ny", "pred_nz",
            "target_nx", "target_ny", "target_nz", "angular_error_deg", "axis_error_deg",
            "point_consensus", "mean_vector_norm",
        ])
        for row in rows:
            writer.writerow([
                row["file"], row["car_model"], *row["pred_normal"], *row["target_normal"],
                row["angular_error_deg"], row["axis_error_deg"], row["point_consensus"], row["mean_vector_norm"],
            ])
    return json_path, csv_path, report["summary"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="MSECNet split inference and held-out-set evaluation")
    parser.add_argument("checkpoint", type=Path, nargs="?", default=DEFAULT_CHECKPOINT)
    parser.add_argument("labels", type=Path, nargs="?", default=DEFAULT_DATASET_DIR / "labels_manual3d.npz")
    parser.add_argument("pcd_dir", type=Path, nargs="?", default=DEFAULT_DATASET_DIR / "clouds")
    parser.add_argument("--centers", type=Path, default=DEFAULT_DATASET_DIR / "anchors_manual3d.json")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--split-name", choices=("train", "val", "test"), default="test")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=None,
                        help="maximum points per cloud; defaults to the checkpoint value (0 keeps all)")
    parser.add_argument("--npoints", type=int, default=None,
                        help="deprecated alias for --max-points")
    parser.add_argument("--ball-radius-m", type=float, default=None,
                        help="sphere radius; defaults to the checkpoint value")
    parser.add_argument("--radial-weight-beta", type=float, default=None,
                        help="pooling weight beta; defaults to the checkpoint value")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this MSECNet pointops implementation")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("--batch-size must be >= 1 and --workers must be >= 0")
    for path in (args.checkpoint, args.labels, args.pcd_dir, args.split):
        if not path.exists():
            raise FileNotFoundError(path)

    centers_path = args.centers or (args.labels.parent / "anchors_manual3d.json")
    if not centers_path.is_file():
        raise FileNotFoundError(centers_path)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"{args.checkpoint} is missing model weights")
    if checkpoint.get("normal_convention") != "oriented_toward_camera":
        raise ValueError("checkpoint is not an oriented center-ball MSECNet checkpoint")
    if "ball_radius_m" not in checkpoint or "radial_weight_beta" not in checkpoint:
        raise ValueError("checkpoint lacks required center-ball metadata")
    ball_radius_m = args.ball_radius_m if args.ball_radius_m is not None else float(checkpoint["ball_radius_m"])
    radial_weight_beta = (
        args.radial_weight_beta if args.radial_weight_beta is not None else float(checkpoint["radial_weight_beta"])
    )
    if ball_radius_m <= 0 or radial_weight_beta < 0:
        raise ValueError("--ball-radius-m must be positive and --radial-weight-beta must be >= 0")
    max_points = resolve_max_points(
        args.max_points, args.npoints, default=int(checkpoint.get("max_points", checkpoint.get("npoints", 1024)))
    )

    labels = np.load(args.labels)
    indices = load_split_indices(labels["files"], args.split, args.split_name)
    files = labels["files"][indices]
    normals = labels["normal"][indices]
    with open(centers_path, encoding="utf-8") as f:
        centers = json.load(f)
    models = load_models(args.labels.parent / "manifest.jsonl")
    dataset = BallNormalDS(
        files, normals, str(args.pcd_dir), max_points, False, centers, ball_radius_m,
        seed=checkpoint.get("seed"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=args.device.startswith("cuda"), collate_fn=collate_variable_points)

    cfg = config.load_cfg_from_cfg_file(os.path.join(HERE, "config", "msecnet_ball.yaml"))
    cfg.num_classes = 3
    model = MSECNet(cfg).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    print(f"loaded checkpoint step={checkpoint.get('step', 'unknown')} max_points={max_points}", flush=True)
    print(f"infer split={args.split_name} samples={len(dataset)} batch_size={args.batch_size}", flush=True)
    rows = infer(model, loader, args.device, files, models, radial_weight_beta)

    out_dir = args.out or (args.checkpoint.parent / f"inference_{args.split_name}")
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "labels": str(args.labels.resolve()),
        "pcd_dir": str(args.pcd_dir.resolve()),
        "split": str(args.split.resolve()),
        "split_name": args.split_name,
        "max_points": max_points,
        "ball_radius_m": ball_radius_m,
        "radial_weight_beta": radial_weight_beta,
        "device": args.device,
        "normal_convention": checkpoint["normal_convention"],
        "aggregation": checkpoint["aggregation"],
        "pooling": checkpoint.get("pooling"),
    }
    json_path, csv_path, summary = write_outputs(out_dir, metadata, rows)
    print(
        "result "
        f"mean={summary['mean_angular_error_deg']:.3f}deg "
        f"median={summary['median_angular_error_deg']:.3f}deg "
        f"<=10deg={summary['acc10_pct']:.1f}%",
        flush=True,
    )
    print(f"report={json_path}", flush=True)
    print(f"predictions={csv_path}", flush=True)


if __name__ == "__main__":
    main()
