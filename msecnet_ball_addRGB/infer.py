#!/usr/bin/env python3
"""Evaluate an RGB-fusion checkpoint on a named existing ball-dataset split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:  # Support both ``python -m`` and direct script execution from the project root.
    from .data import BallRGBNormalDataset, collate_ball_rgb, source_images_from_index
    from .model import normalized
    from .train import build_model
except ImportError:
    from data import BallRGBNormalDataset, collate_ball_rgb, source_images_from_index
    from model import normalized
    from train import build_model


def split_index(files, split_path: Path, name: str):
    split = json.loads(split_path.read_text(encoding="utf-8"))
    selected = set(split.get(name, []))
    if not selected:
        raise ValueError(f"split {name!r} is empty or absent")
    names = [str(item) for item in files]
    unknown = selected - set(names)
    if unknown:
        raise ValueError(f"split contains {len(unknown)} unknown files")
    return np.array([index for index, value in enumerate(names) if value in selected], dtype=np.int64)


@torch.no_grad()
def run(model, loader, device, beta):
    model.eval(); rows = []
    for coord, radial, offset, image, target, _, names, counts in loader:
        output = model(coord.to(device), radial.to(device), offset.to(device), image.to(device), counts, beta)
        target = target.to(device)
        for index, name in enumerate(names):
            predictions = {key: normalized(output[key])[index].cpu().numpy() for key in ("geometry_vector", "image_vector", "fused_vector")}
            cosine = float(np.clip(np.dot(predictions["fused_vector"], target[index].cpu().numpy()), -1, 1))
            rows.append({
                "file": name, "target_normal": target[index].cpu().tolist(),
                "geometry_normal": predictions["geometry_vector"].tolist(), "image_normal": predictions["image_vector"].tolist(),
                "pred_normal": predictions["fused_vector"].tolist(), "angular_error_deg": float(np.degrees(np.arccos(cosine))),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate a late RGB/point-cloud fusion checkpoint")
    parser.add_argument("checkpoint", type=Path); parser.add_argument("labels", type=Path); parser.add_argument("pcd_dir", type=Path)
    parser.add_argument("--source-root", type=Path, required=True); parser.add_argument("--centers", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True); parser.add_argument("--split-name", choices=("train", "val", "test"), default="test")
    parser.add_argument("--out", type=Path, default=None); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by MSECNet pointops")
    for path in (args.checkpoint, args.labels, args.pcd_dir, args.source_root, args.centers, args.split):
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    schema = checkpoint.get("input_schema", {})
    if schema.get("fusion") != "late global feature fusion":
        raise ValueError("checkpoint is not an msecnet_ball_addRGB late-fusion checkpoint")
    settings = checkpoint.get("args", {})
    labels = np.load(args.labels); index = split_index(labels["files"], args.split, args.split_name)
    anchors = json.loads(args.centers.read_text(encoding="utf-8"))
    dataset = BallRGBNormalDataset(
        labels["files"][index], labels["normal"][index], args.pcd_dir, anchors, source_images_from_index(args.source_root),
        float(checkpoint["ball_radius_m"]), int(settings.get("max_points", 1024)), int(settings.get("image_size", 160)),
        float(settings.get("image_crop_scale", 2.5)), False, seed=int(settings.get("seed", 0)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, collate_fn=collate_ball_rgb)
    model = build_model(False).to(args.device); model.load_state_dict(checkpoint["model"], strict=True)
    rows = run(model, loader, args.device, float(checkpoint["radial_weight_beta"]))
    errors = np.asarray([row["angular_error_deg"] for row in rows])
    summary = {"samples": len(rows), "mean_angular_error_deg": float(errors.mean()), "median_angular_error_deg": float(np.median(errors)),
               "acc10_pct": float((errors <= 10).mean() * 100)}
    out_dir = args.out or args.checkpoint.parent / f"inference_{args.split_name}"; out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"checkpoint": str(args.checkpoint.resolve()), "split_name": args.split_name, "source_root": str(args.source_root.resolve()),
                "normal_convention": checkpoint["normal_convention"], "input_schema": schema}
    (out_dir / "report.json").write_text(json.dumps({"metadata": metadata, "summary": summary, "predictions": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["file", "pred_nx", "pred_ny", "pred_nz", "target_nx", "target_ny", "target_nz", "angular_error_deg"])
        for row in rows:
            writer.writerow([row["file"], *row["pred_normal"], *row["target_normal"], row["angular_error_deg"]])
    print(f"result split={args.split_name} mean={summary['mean_angular_error_deg']:.3f}deg median={summary['median_angular_error_deg']:.3f}deg "
          f"<=10deg={summary['acc10_pct']:.1f}% report={out_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
