#!/usr/bin/env python3
"""Predict normals for unlabeled OBB crops with a center-ball MSECNet.

The unlabeled OBB dataset has no reviewed 3D center. Its crop centroid is used
as a deterministic proxy center, then the model receives the same fixed-radius
ball coordinates, radial feature, and weighted pooling as during ball training.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MROOT = PROJECT_ROOT / "msecnet" / "MSECNet"
sys.path.insert(0, str(MROOT / "model"))
sys.path.insert(0, str(MROOT / "scripts"))

from architectures import MSECNet  # noqa: E402
from util import config  # noqa: E402
from train import (  # noqa: E402
    aggregate_point_normals,
    radial_weights,
    resolve_max_points,
    to_msecnet,
)


PLY_DTYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
}


def load_ply_xyz(path):
    """Load XYZ from the binary little-endian source clouds used by the open set."""
    with open(path, "rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        vertex_count, properties, in_vertex = None, [], False
        while True:
            line = stream.readline().decode("ascii").strip()
            if line == "end_header":
                break
            fields = line.split()
            if fields[:2] == ["element", "vertex"]:
                vertex_count, in_vertex = int(fields[2]), True
            elif fields[:1] == ["element"]:
                in_vertex = False
            elif in_vertex and fields[:1] == ["property"]:
                if len(fields) != 3 or fields[1] not in PLY_DTYPES:
                    raise ValueError(f"unsupported PLY vertex property: {line}")
                properties.append((fields[2], "<" + PLY_DTYPES[fields[1]]))
        if vertex_count is None or not {"x", "y", "z"}.issubset(dict(properties)):
            raise ValueError(f"PLY has no XYZ vertex fields: {path}")
        vertices = np.fromfile(stream, dtype=np.dtype(properties), count=vertex_count)
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)


class UnlabeledBallDataset(Dataset):
    """Use an OBB centroid as center, then take the ball from the source cloud."""

    def __init__(self, files, cloud_dir, source_paths, max_points, ball_radius_m, min_ball_points):
        self.files = [str(name) for name in files]
        self.cloud_dir = Path(cloud_dir)
        self.source_paths = source_paths
        self.max_points = max_points
        self.ball_radius_m = ball_radius_m
        self.min_ball_points = min_ball_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.cloud_dir / self.files[index]
        with np.load(path) as data:
            xyz = data["xyz"]
            if "label" in data:
                xyz = xyz[data["label"] == 1]
        if not len(xyz):
            raise ValueError(f"{path} has no usable points")
        center = xyz.astype(np.float32).mean(axis=0).astype(np.float32)
        source_path = self.source_paths.get(self.files[index])
        if source_path is None or not source_path.is_file():
            raise FileNotFoundError(f"missing source cloud for {self.files[index]}: {source_path}")
        xyz = load_ply_xyz(source_path)
        xyz = xyz[np.linalg.norm(xyz - center, axis=1) <= self.ball_radius_m]
        if len(xyz) < self.min_ball_points:
            raise ValueError(
                f"{path} has {len(xyz)} points in the centroid-centered {self.ball_radius_m} m ball; "
                f"need at least {self.min_ball_points}"
            )
        source_ball_points = len(xyz)
        if self.max_points and len(xyz) > self.max_points:
            keep = np.random.default_rng(index).choice(len(xyz), self.max_points, replace=False)
            xyz = xyz[keep]
        coord = (xyz - center) / self.ball_radius_m
        radial_distance = np.minimum(np.linalg.norm(coord, axis=1, keepdims=True), 1.0)
        return (
            coord.astype(np.float32), radial_distance.astype(np.float32), self.files[index], center,
            np.int32(source_ball_points),
        )


def collate_unlabeled_ball(batch):
    coords = [torch.from_numpy(item[0]) for item in batch]
    radial_distances = [torch.from_numpy(item[1]) for item in batch]
    counts = torch.tensor([len(coord) for coord in coords], dtype=torch.int64)
    return (
        torch.cat(coords).float(), torch.cat(radial_distances).float(),
        torch.cumsum(counts, dim=0).int(), [item[2] for item in batch],
        torch.from_numpy(np.stack([item[3] for item in batch])).float(),
        torch.tensor([item[4] for item in batch], dtype=torch.int32), counts,
    )


@torch.no_grad()
def predict(model, loader, device, radial_weight_beta):
    model.eval()
    rows = []
    for coord, radial_distance, offset, names, centers, ball_counts, counts in loader:
        coord, feat, offset = to_msecnet(
            coord.to(device, non_blocking=True), radial_distance.to(device, non_blocking=True),
            offset.to(device, non_blocking=True),
        )
        point_vectors = model(coord, feat, offset)
        mean_vector, normal, _ = aggregate_point_normals(
            point_vectors, counts, radial_weights(feat, radial_weight_beta)
        )
        for index, name in enumerate(names):
            prediction = normal[index].cpu().numpy()
            center = centers[index].numpy()
            if float(np.dot(prediction, -center)) < 0:
                prediction = -prediction
            rows.append({
                "file": name,
                "pred_normal": [float(value) for value in prediction.tolist()],
                "center_3d": [float(value) for value in center.tolist()],
                "ball_points": int(ball_counts[index]),
                "point_consensus": float(mean_vector[index].norm().cpu()),
                "mean_vector_norm": float(mean_vector[index].norm().cpu()),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Center-ball MSECNet prediction for unlabeled OBB crops")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="root that contains each manifest source_cloud PLY")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--npoints", type=int, default=None)
    parser.add_argument("--ball-radius-m", type=float, default=None)
    parser.add_argument("--radial-weight-beta", type=float, default=None)
    parser.add_argument("--min-ball-points", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    files_path = args.dataset_dir / "unlabeled_test.npz"
    cloud_dir = args.dataset_dir / "clouds"
    manifest_path = args.dataset_dir / "manifest.jsonl"
    for path in (args.checkpoint, files_path, cloud_dir, manifest_path, args.source_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this MSECNet pointops implementation")
    if args.batch_size < 1 or args.workers < 0 or args.min_ball_points < 1:
        raise ValueError("--batch-size and --min-ball-points must be positive; --workers must be >= 0")

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

    files = np.load(files_path)["files"]
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    source_paths = {
        str(item["file"]): args.source_root / item["source_cloud"]
        for item in manifest if "file" in item and "source_cloud" in item
    }
    missing_sources = [str(name) for name in files if str(name) not in source_paths]
    if missing_sources:
        raise ValueError(f"manifest has no source_cloud for {len(missing_sources)} samples")
    dataset = UnlabeledBallDataset(
        files, cloud_dir, source_paths, max_points, ball_radius_m, args.min_ball_points,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_unlabeled_ball,
    )
    cfg = config.load_cfg_from_cfg_file(str(HERE / "config" / "msecnet_ball.yaml"))
    cfg.num_classes = 3
    model = MSECNet(cfg).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    print(
        f"loaded checkpoint step={checkpoint.get('step', 'unknown')} max_points={max_points} "
        f"ball_radius_m={ball_radius_m}", flush=True,
    )
    print(f"predict unlabeled samples={len(dataset)} batch_size={args.batch_size}", flush=True)
    rows = predict(model, loader, args.device, radial_weight_beta)

    out_dir = args.out or (args.checkpoint.parent / "unlabeled_predictions" / args.dataset_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "source_root": str(args.source_root.resolve()),
        "unlabeled": True,
        "normal_convention": "oriented_toward_camera",
        "center_definition": "mean of the unlabeled OBB crop points; ball points from source_cloud",
        "ball_radius_m": ball_radius_m,
        "radial_weight_beta": radial_weight_beta,
        "pooling": checkpoint.get("pooling"),
        "max_points": max_points,
        "device": args.device,
    }
    report = {"metadata": metadata, "summary": {"samples": len(rows)}, "predictions": rows}
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "center_x", "center_y", "center_z", "pred_nx", "pred_ny", "pred_nz",
            "ball_points", "point_consensus", "mean_vector_norm",
        ])
        for row in rows:
            writer.writerow([
                row["file"], *row["center_3d"], *row["pred_normal"], row["ball_points"],
                row["point_consensus"], row["mean_vector_norm"],
            ])
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
