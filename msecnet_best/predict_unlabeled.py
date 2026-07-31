#!/usr/bin/env python3
"""Predict cap normals for an unlabeled, already-cropped MSECNet dataset."""
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
MROOT = HERE / "MSECNet"
sys.path.insert(0, str(MROOT / "model"))
sys.path.insert(0, str(MROOT / "scripts"))

from architectures import MSECNet  # noqa: E402
from util import config  # noqa: E402
from train import aggregate_point_normals, resolve_max_points, to_msecnet  # noqa: E402


class UnlabeledPatchDataset(Dataset):
    def __init__(self, files, cloud_dir, max_points):
        self.files = [str(name) for name in files]
        self.cloud_dir = Path(cloud_dir)
        self.max_points = max_points

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
        xyz = xyz.astype(np.float32)
        if self.max_points and len(xyz) > self.max_points:
            xyz = xyz[np.random.default_rng(index).choice(len(xyz), self.max_points, replace=False)]
        center = xyz.mean(axis=0).astype(np.float32)
        xyz -= center
        xyz /= np.linalg.norm(xyz, axis=1).max() + 1e-9
        return xyz, self.files[index], center


def collate_unlabeled(batch):
    coords = [torch.from_numpy(item[0]) for item in batch]
    counts = torch.tensor([len(coord) for coord in coords], dtype=torch.int64)
    centers = torch.from_numpy(np.stack([item[2] for item in batch])).float()
    return torch.cat(coords).float(), torch.cumsum(counts, dim=0).int(), [item[1] for item in batch], centers, counts


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    rows = []
    for coord, offset, names, centers, counts in loader:
        coord, feat, offset = to_msecnet(coord.to(device, non_blocking=True), offset.to(device, non_blocking=True))
        mean_vector, normal = aggregate_point_normals(model(coord, feat, offset), counts)
        for index, name in enumerate(names):
            prediction = normal[index].cpu().numpy()
            # Camera origin is (0, 0, 0); require the output to point toward it.
            if float(np.dot(prediction, -centers[index].numpy())) < 0:
                prediction = -prediction
            rows.append({
                "file": name,
                "pred_normal": [float(value) for value in prediction.tolist()],
                "mean_vector_norm": float(mean_vector[index].norm().cpu()),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="MSECNet prediction for unlabeled OBB-cropped test data")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--npoints", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    files_path = args.dataset_dir / "unlabeled_test.npz"
    cloud_dir = args.dataset_dir / "clouds"
    for path in (args.checkpoint, files_path, cloud_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this MSECNet pointops implementation")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("--batch-size must be >= 1 and --workers must be >= 0")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"{args.checkpoint} is missing model weights")
    max_points = resolve_max_points(
        args.max_points, args.npoints, default=int(checkpoint.get("max_points", checkpoint.get("npoints", 1024)))
    )
    files = np.load(files_path)["files"]
    dataset = UnlabeledPatchDataset(files, cloud_dir, max_points)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, collate_fn=collate_unlabeled)

    cfg = config.load_cfg_from_cfg_file(str(MROOT / "scripts" / "config" / "pcpnet" / "config.yaml"))
    cfg.num_classes = 3
    model = MSECNet(cfg).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    print(f"loaded checkpoint step={checkpoint.get('step', 'unknown')} max_points={max_points}", flush=True)
    print(f"predict unlabeled samples={len(dataset)} batch_size={args.batch_size}", flush=True)
    rows = predict(model, loader, args.device)

    out_dir = args.out or (args.checkpoint.parent / "unlabeled_predictions" / args.dataset_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": checkpoint.get("step"),
        "dataset_dir": str(args.dataset_dir.resolve()), "unlabeled": True,
        "normal_convention": "toward_camera",
        "max_points": max_points, "device": args.device,
    }
    report = {"metadata": metadata, "summary": {"samples": len(rows)}, "predictions": rows}
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "pred_nx", "pred_ny", "pred_nz", "mean_vector_norm"])
        for row in rows:
            writer.writerow([row["file"], *row["pred_normal"], row["mean_vector_norm"]])
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
