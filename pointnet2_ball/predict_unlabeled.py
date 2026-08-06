#!/usr/bin/env python3
"""Run a PointNet++ ball checkpoint on unlabeled OBB crops."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:  # Package import for tests; script import for the documented commands.
    from .data import resolve_num_points, sample_fixed_points
    from .infer import load_checkpoint
except ImportError:  # pragma: no cover - exercised when invoked as a file
    from data import resolve_num_points, sample_fixed_points
    from infer import load_checkpoint


PLY_DTYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
    "float": "f4", "double": "f8",
}


def load_ply_xyz(path):
    """Load XYZ from the project's binary little-endian source PLY clouds."""
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
    """Use the OBB crop centroid as a proxy center, exactly as msecnet_ball does."""

    def __init__(self, files, cloud_dir, source_paths, num_points, ball_radius_m, min_ball_points):
        self.files = [str(name) for name in files]
        self.cloud_dir, self.source_paths = Path(cloud_dir), source_paths
        self.num_points, self.ball_radius_m, self.min_ball_points = num_points, ball_radius_m, min_ball_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = self.files[index]
        with np.load(self.cloud_dir / file_name, allow_pickle=False) as cloud:
            crop = cloud["xyz"].astype(np.float32)
            if "label" in cloud:
                crop = crop[cloud["label"] == 1]
        if not len(crop):
            raise ValueError(f"{file_name} has no usable OBB crop points")
        center = crop.mean(axis=0, dtype=np.float32)
        source_path = self.source_paths.get(file_name)
        if source_path is None or not source_path.is_file():
            raise FileNotFoundError(f"missing source cloud for {file_name}: {source_path}")
        ball = load_ply_xyz(source_path)
        ball = ball[np.linalg.norm(ball - center, axis=1) <= self.ball_radius_m]
        if len(ball) < self.min_ball_points:
            raise ValueError(f"{file_name} has {len(ball)} proxy-ball points; need {self.min_ball_points}")
        source_count = len(ball)
        ball = sample_fixed_points(ball, self.num_points, np.random.default_rng(index))
        xyz = (ball - center) / self.ball_radius_m
        radial = np.minimum(np.linalg.norm(xyz, axis=1, keepdims=True), 1.0)
        return xyz.astype(np.float32), radial.astype(np.float32), file_name, center, np.int32(source_count)


def collate_unlabeled(batch):
    return (
        torch.from_numpy(np.stack([item[0] for item in batch])).float(),
        torch.from_numpy(np.stack([item[1] for item in batch])).float(),
        [item[2] for item in batch], torch.from_numpy(np.stack([item[3] for item in batch])).float(),
        torch.tensor([item[4] for item in batch], dtype=torch.int32),
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    rows = []
    for xyz, radial, names, centers, counts in loader:
        normals = model.predict_normal(xyz.to(device), radial.to(device)).cpu().numpy()
        for index, name in enumerate(names):
            normal, center = normals[index], centers[index].numpy()
            # The supervised convention is towards the camera at coordinate origin.
            if float(np.dot(normal, -center)) < 0:
                normal = -normal
            rows.append({
                "file": name, "pred_normal": [float(value) for value in normal],
                "center_3d": [float(value) for value in center], "ball_points": int(counts[index]),
            })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="PointNet++ center-ball prediction for unlabeled OBB crops")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--npoints", type=int, default=None, help="deprecated alias for --num-points")
    parser.add_argument("--ball-radius-m", type=float, default=None)
    parser.add_argument("--min-ball-points", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0 or args.min_ball_points < 1:
        raise ValueError("invalid batch size, worker count, or minimum ball point count")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable; pass --device cpu explicitly")
    files_path, cloud_dir, manifest_path = args.dataset_dir / "unlabeled_test.npz", args.dataset_dir / "clouds", args.dataset_dir / "manifest.jsonl"
    for path in (args.checkpoint, files_path, cloud_dir, manifest_path, args.source_root):
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint, model = load_checkpoint(args.checkpoint, args.device)
    num_points = resolve_num_points(args.num_points, args.npoints, default=int(checkpoint["num_points"]))
    radius = float(checkpoint["ball_radius_m"]) if args.ball_radius_m is None else args.ball_radius_m
    if radius <= 0:
        raise ValueError("--ball-radius-m must be positive")
    files = np.load(files_path, allow_pickle=False)["files"]
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    source_paths = {str(row["file"]): args.source_root / row["source_cloud"] for row in manifest if "file" in row and "source_cloud" in row}
    if missing := [str(name) for name in files if str(name) not in source_paths]:
        raise ValueError(f"manifest has no source_cloud for {len(missing)} samples")
    dataset = UnlabeledBallDataset(files, cloud_dir, source_paths, num_points, radius, args.min_ball_points)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=args.device.startswith("cuda"), collate_fn=collate_unlabeled)
    rows = predict(model, loader, args.device)
    output_dir = args.out or (args.checkpoint.parent / "unlabeled_predictions" / args.dataset_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": {
            "checkpoint": str(args.checkpoint.resolve()), "dataset_dir": str(args.dataset_dir.resolve()),
            "source_root": str(args.source_root.resolve()), "unlabeled": True,
            "normal_convention": "oriented_toward_camera", "ball_radius_m": radius,
            "num_points": num_points, "device": args.device,
        },
        "summary": {"samples": len(rows)}, "predictions": rows,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(output_dir / "predictions.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "center_x", "center_y", "center_z", "pred_nx", "pred_ny", "pred_nz", "ball_points"])
        for row in rows:
            writer.writerow([row["file"], *row["center_3d"], *row["pred_normal"], row["ball_points"]])
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
