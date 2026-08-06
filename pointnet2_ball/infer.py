#!/usr/bin/env python3
"""Evaluate a PointNet++ center-ball checkpoint on train, val, or test."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:  # Package import for tests; script import for the documented commands.
    from .data import BallNormalDataset, collate_fixed_points, load_split_indices, oriented_angular_error_deg, resolve_num_points
    from .model import build_model
except ImportError:  # pragma: no cover - exercised when invoked as a file
    from data import BallNormalDataset, collate_fixed_points, load_split_indices, oriented_angular_error_deg, resolve_num_points
    from model import build_model


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATASET = ROOT / "data" / "msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08"


def load_car_models(manifest_path):
    if not manifest_path.is_file():
        return {}
    return {
        row["file"]: row.get("car_model", "")
        for row in (json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line)
    }


def summarize(rows):
    errors = np.asarray([row["angular_error_deg"] for row in rows], dtype=np.float64)
    return {
        "samples": int(len(rows)),
        "mean_angular_error_deg": float(errors.mean()),
        "median_angular_error_deg": float(np.median(errors)),
        "acc10_pct": float((errors <= 10).mean() * 100),
        # Kept for the existing report viewer schema; these remain directed errors.
        "mean_axis_error_deg": float(errors.mean()),
        "median_axis_error_deg": float(np.median(errors)),
    }


@torch.no_grad()
def infer(model, loader, device, files, car_models):
    model.eval()
    rows, offset = [], 0
    for xyz, radial, target, _ in loader:
        xyz, radial, target = xyz.to(device), radial.to(device), target.to(device)
        prediction = model.predict_normal(xyz, radial)
        errors = oriented_angular_error_deg(prediction, target)
        for index in range(len(xyz)):
            file_name = str(files[offset + index])
            rows.append({
                "file": file_name,
                "car_model": car_models.get(file_name, ""),
                "pred_normal": [float(value) for value in prediction[index].cpu().tolist()],
                "target_normal": [float(value) for value in target[index].cpu().tolist()],
                "angular_error_deg": float(errors[index].cpu()),
                "axis_error_deg": float(errors[index].cpu()),
                # PointNet++ has a single global head, not per-point votes.
                "point_consensus": None,
                "mean_vector_norm": 1.0,
            })
        offset += len(xyz)
    if offset != len(files):
        raise RuntimeError(f"inference returned {offset} samples; expected {len(files)}")
    return rows


def write_outputs(output_dir, metadata, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["car_model"]].append(row)
    report = {
        "metadata": metadata,
        "summary": summarize(rows),
        "by_car_model": {name: summarize(items) for name, items in sorted(by_model.items())},
        "predictions": rows,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(output_dir / "predictions.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "file", "car_model", "pred_nx", "pred_ny", "pred_nz", "target_nx", "target_ny", "target_nz",
            "angular_error_deg", "axis_error_deg", "point_consensus", "mean_vector_norm",
        ])
        for row in rows:
            writer.writerow([
                row["file"], row["car_model"], *row["pred_normal"], *row["target_normal"],
                row["angular_error_deg"], row["axis_error_deg"], "", row["mean_vector_norm"],
            ])
    return report_path, output_dir / "predictions.csv", report["summary"]


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "pointnet2_global_normal_v1" or "model" not in checkpoint:
        raise ValueError("checkpoint is not a PointNet++ center-ball normal checkpoint")
    if checkpoint.get("normal_convention") != "oriented_toward_camera":
        raise ValueError("checkpoint does not use camera-oriented normals")
    required = ("ball_radius_m", "num_points", "model_args")
    if any(name not in checkpoint for name in required):
        raise ValueError("checkpoint lacks PointNet++ ball metadata")
    model = build_model(checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint, model


def main(argv=None):
    parser = argparse.ArgumentParser(description="PointNet++ split inference for center-ball normals")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("labels", type=Path, nargs="?", default=DEFAULT_DATASET / "labels_manual3d.npz")
    parser.add_argument("pcd_dir", type=Path, nargs="?", default=DEFAULT_DATASET / "clouds")
    parser.add_argument("--centers", type=Path, default=DEFAULT_DATASET / "anchors_manual3d.json")
    parser.add_argument("--split", type=Path, default=DEFAULT_DATASET / "split_by_generalization_group.json")
    parser.add_argument("--split-name", choices=("train", "val", "test"), default="test")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--npoints", type=int, default=None, help="deprecated alias for --num-points")
    parser.add_argument("--ball-radius-m", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("--batch-size must be positive and --workers must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable; pass --device cpu explicitly")
    for path in (args.checkpoint, args.labels, args.pcd_dir, args.centers, args.split):
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint, model = load_checkpoint(args.checkpoint, args.device)
    num_points = resolve_num_points(args.num_points, args.npoints, default=int(checkpoint["num_points"]))
    radius = float(checkpoint["ball_radius_m"]) if args.ball_radius_m is None else args.ball_radius_m
    if radius <= 0:
        raise ValueError("--ball-radius-m must be positive")
    labels = np.load(args.labels, allow_pickle=False)
    indices = load_split_indices(labels["files"], args.split, args.split_name)
    files, normals = labels["files"][indices], labels["normal"][indices]
    anchors = json.loads(args.centers.read_text(encoding="utf-8"))
    dataset = BallNormalDataset(files, normals, args.pcd_dir, anchors, radius, num_points, train=False, seed=checkpoint.get("seed", 20260722))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=args.device.startswith("cuda"), collate_fn=collate_fixed_points)
    print(f"loaded PointNet++ checkpoint step={checkpoint.get('step', 'unknown')} points={num_points}", flush=True)
    print(f"infer split={args.split_name} samples={len(dataset)} batch_size={args.batch_size}", flush=True)
    rows = infer(model, loader, args.device, files, load_car_models(args.labels.parent / "manifest.jsonl"))
    output_dir = args.out or (args.checkpoint.parent / f"inference_{args.split_name}")
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": checkpoint.get("step"),
        "labels": str(args.labels.resolve()), "pcd_dir": str(args.pcd_dir.resolve()), "split": str(args.split.resolve()),
        "split_name": args.split_name, "num_points": num_points, "ball_radius_m": radius, "device": args.device,
        "normal_convention": checkpoint["normal_convention"], "aggregation": checkpoint["aggregation"],
        "pooling": checkpoint["pooling"],
    }
    report_path, csv_path, summary = write_outputs(output_dir, metadata, rows)
    print(f"result mean={summary['mean_angular_error_deg']:.3f}deg median={summary['median_angular_error_deg']:.3f}deg <=10deg={summary['acc10_pct']:.1f}%", flush=True)
    print(f"report={report_path}\npredictions={csv_path}", flush=True)


if __name__ == "__main__":
    main()
