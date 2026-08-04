#!/usr/bin/env python3
"""Evaluate MSECNet normal predictions on a named dataset split.

Without arguments, this evaluates the completed Manual Pseudo-OBB training run
on its held-out test split. The dataset and checkpoint paths may be overridden
through the positional path arguments and options below.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
MROOT = os.path.join(HERE, "MSECNet")
PROJECT_ROOT = Path(HERE).parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "msecnet_best_fuelcap_pass_20260803_10847_manual3d_pseudo_obb"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "msecnet_best" / "out" / "pseudo_obb_20260803_v1" / "best.pt"
sys.path.insert(0, os.path.join(MROOT, "model"))
sys.path.insert(0, os.path.join(MROOT, "scripts"))

from architectures import MSECNet  # noqa: E402
from util import config  # noqa: E402
from train import (  # noqa: E402
    CapNormalDS,
    aggregate_point_normals,
    collate_variable_points,
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
    errors = np.array([row["axis_error_deg"] for row in rows], dtype=np.float64)
    return {
        "samples": int(len(rows)),
        "mean_axis_error_deg": float(errors.mean()),
        "median_axis_error_deg": float(np.median(errors)),
        "acc10_pct": float((errors <= 10.0).mean() * 100.0),
    }


@torch.no_grad()
def infer(model, loader, device, file_names, model_by_file):
    model.eval()
    rows = []
    sample_offset = 0
    for coord, offset, target, _, counts in loader:
        coord, feat, offset = to_msecnet(
            coord.to(device, non_blocking=True), offset.to(device, non_blocking=True)
        )
        mean_vector, normal = aggregate_point_normals(model(coord, feat, offset), counts)
        target = target.to(device, non_blocking=True)
        cosine = (normal * target).sum(1).abs().clamp(0, 1)
        errors = torch.rad2deg(torch.arccos(cosine))
        confidence = mean_vector.norm(dim=1)
        for batch_index in range(len(counts)):
            name = str(file_names[sample_offset + batch_index])
            rows.append({
                "file": name,
                "car_model": model_by_file.get(name, ""),
                "pred_normal": [float(v) for v in normal[batch_index].cpu().tolist()],
                "target_normal": [float(v) for v in target[batch_index].cpu().tolist()],
                "axis_error_deg": float(errors[batch_index].cpu()),
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
            "target_nx", "target_ny", "target_nz", "axis_error_deg", "mean_vector_norm",
        ])
        for row in rows:
            writer.writerow([
                row["file"], row["car_model"], *row["pred_normal"], *row["target_normal"],
                row["axis_error_deg"], row["mean_vector_norm"],
            ])
    return json_path, csv_path, report["summary"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="MSECNet split inference and held-out-set evaluation")
    parser.add_argument("checkpoint", type=Path, nargs="?", default=DEFAULT_CHECKPOINT)
    parser.add_argument("labels", type=Path, nargs="?", default=DEFAULT_DATASET_DIR / "labels_manual3d.npz")
    parser.add_argument("pcd_dir", type=Path, nargs="?", default=DEFAULT_DATASET_DIR / "clouds")
    parser.add_argument("--centers", type=Path, default=DEFAULT_DATASET_DIR / "anchors_manual3d.json")
    parser.add_argument("--split", type=Path, default=DEFAULT_DATASET_DIR / "split_by_car_model.json")
    parser.add_argument("--split-name", choices=("train", "val", "test"), default="test")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=None,
                        help="maximum points per cloud; defaults to the checkpoint value (0 keeps all)")
    parser.add_argument("--npoints", type=int, default=None,
                        help="deprecated alias for --max-points")
    parser.add_argument("--radius", type=float, default=0.3)
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
    dataset = CapNormalDS(files, normals, str(args.pcd_dir), max_points, False, centers, args.radius)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=args.device.startswith("cuda"), collate_fn=collate_variable_points)

    cfg = config.load_cfg_from_cfg_file(os.path.join(MROOT, "scripts/config/pcpnet/config.yaml"))
    cfg.num_classes = 3
    model = MSECNet(cfg).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    print(f"loaded checkpoint step={checkpoint.get('step', 'unknown')} max_points={max_points}", flush=True)
    print(f"infer split={args.split_name} samples={len(dataset)} batch_size={args.batch_size}", flush=True)
    rows = infer(model, loader, args.device, files, models)

    out_dir = args.out or (args.checkpoint.parent / f"inference_{args.split_name}")
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "labels": str(args.labels.resolve()),
        "pcd_dir": str(args.pcd_dir.resolve()),
        "split": str(args.split.resolve()),
        "split_name": args.split_name,
        "max_points": max_points,
        "radius": args.radius,
        "device": args.device,
    }
    json_path, csv_path, summary = write_outputs(out_dir, metadata, rows)
    print(
        "result "
        f"mean={summary['mean_axis_error_deg']:.3f}deg "
        f"median={summary['median_axis_error_deg']:.3f}deg "
        f"<=10deg={summary['acc10_pct']:.1f}%",
        flush=True,
    )
    print(f"report={json_path}", flush=True)
    print(f"predictions={csv_path}", flush=True)


if __name__ == "__main__":
    main()
