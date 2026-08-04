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
    from .data import BallRGBNormalDataset, collate_ball_rgb, load_obb_detections, source_images_from_index
    from .model import normalized
    from .train import build_model
except ImportError:
    from data import BallRGBNormalDataset, collate_ball_rgb, load_obb_detections, source_images_from_index
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
    for coord, radial, offset, image, target, _, names, counts, point_uv in loader:
        output = model(coord.to(device), radial.to(device), offset.to(device), image.to(device), counts, beta,
                       point_uv=point_uv.to(device))
        target = target.to(device)
        for index, name in enumerate(names):
            predictions = {key: normalized(output[key])[index].cpu().numpy() for key in ("geometry_vector", "image_vector", "fused_vector")}
            target_normal = target[index].cpu().numpy()
            branch_errors = {
                key: float(np.degrees(np.arccos(np.clip(np.dot(value, target_normal), -1, 1))))
                for key, value in predictions.items()
            }
            rows.append({
                "file": name, "target_normal": target_normal.tolist(),
                "geometry_normal": predictions["geometry_vector"].tolist(), "image_normal": predictions["image_vector"].tolist(),
                "pred_normal": predictions["fused_vector"].tolist(),
                "geometry_angular_error_deg": branch_errors["geometry_vector"],
                "image_angular_error_deg": branch_errors["image_vector"],
                "angular_error_deg": branch_errors["fused_vector"],
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate a late RGB/point-cloud fusion checkpoint")
    parser.add_argument("checkpoint", type=Path); parser.add_argument("labels", type=Path); parser.add_argument("pcd_dir", type=Path)
    parser.add_argument("--source-root", type=Path, required=True); parser.add_argument("--centers", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True); parser.add_argument("--split-name", choices=("train", "val", "test"), default="test")
    parser.add_argument("--obb-detections", type=Path, default=None,
                        help="detector OBB cache used to train the checkpoint; required for OBB-crop checkpoints")
    parser.add_argument("--out", type=Path, default=None); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by MSECNet pointops")
    for path in (args.checkpoint, args.labels, args.pcd_dir, args.source_root, args.centers, args.split):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.obb_detections is not None and not args.obb_detections.is_file():
        raise FileNotFoundError(args.obb_detections)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    schema = checkpoint.get("input_schema", {})
    if schema.get("fusion") not in (
        "late global feature fusion", "gated residual global feature fusion",
        "frozen point baseline + gated residual RGB fusion", "point-aligned DINO patch residual RGB fusion",
    ):
        raise ValueError("checkpoint is not an msecnet_ball_addRGB late-fusion checkpoint")
    settings = checkpoint.get("args", {})
    uses_obb = schema.get("rgb_crop_source") == "detector_obb"
    if uses_obb != (args.obb_detections is not None):
        expected = "requires --obb-detections" if uses_obb else "does not accept --obb-detections"
        raise ValueError(f"checkpoint RGB crop protocol {expected}")
    labels = np.load(args.labels); index = split_index(labels["files"], args.split, args.split_name)
    # Match the point-only dataset's original split-local sampling sequence even
    # after OBB-missing samples are filtered out.
    sampling_indices = np.arange(len(index), dtype=np.int64)
    anchors = json.loads(args.centers.read_text(encoding="utf-8"))
    obb_detections = load_obb_detections(args.obb_detections) if args.obb_detections else None
    if obb_detections is not None:
        original_count = len(index)
        keep = np.asarray([str(labels["files"][item]) in obb_detections for item in index], dtype=bool)
        index, sampling_indices = index[keep], sampling_indices[keep]
        print(f"OBB RGB crops: evaluating {len(index)}/{original_count} samples with detections", flush=True)
    if not len(index):
        raise ValueError(f"split {args.split_name!r} has no samples with an OBB detection")
    sampling_seed = checkpoint.get("geometry_sampling_seed", settings.get("geometry_sampling_seed", settings.get("seed", 0)))
    if "geometry_sampling_seed" not in checkpoint and "geometry_sampling_seed" not in settings:
        geometry_path = checkpoint.get("geometry_checkpoint", settings.get("geometry_checkpoint"))
        if geometry_path:
            geometry_path = Path(geometry_path)
            if not geometry_path.is_file():
                geometry_path = Path(__file__).resolve().parent.parent / geometry_path
            if geometry_path.is_file():
                geometry_checkpoint = torch.load(geometry_path, map_location="cpu", weights_only=False)
                sampling_seed = geometry_checkpoint.get("seed", sampling_seed)
    dataset = BallRGBNormalDataset(
        labels["files"][index], labels["normal"][index], args.pcd_dir, anchors, source_images_from_index(args.source_root),
        float(checkpoint["ball_radius_m"]), int(settings.get("max_points", 1024)), int(settings.get("image_size", 160)),
        float(settings.get("image_crop_scale", 2.5)), False, seed=int(settings.get("seed", 0)),
        obb_detections=obb_detections, obb_crop_scale=float(checkpoint.get("obb_crop_scale") or 1.4),
        obb_crop_mode="rectified" if schema.get("rgb") == "rectified detector-OBB source-image crop" else "camera_oriented",
        sampling_indices=sampling_indices, sampling_seed=int(sampling_seed),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, collate_fn=collate_ball_rgb)
    image_backbone = settings.get("image_backbone", "resnet18")
    model = build_model(
        bool(settings.get("image_pretrained", False)), image_backbone=image_backbone,
        dino_unfreeze_blocks=int(settings.get("dino_unfreeze_blocks", 2)),
        fusion_mode=settings.get("fusion_mode", "legacy"),
        geometry_mode=checkpoint.get("geometry_mode", settings.get("geometry_mode", "feature_head")),
        # Missing in V1/V2: retain their original unconstrained residual path.
        max_rgb_correction=checkpoint.get("max_rgb_correction", settings.get("max_rgb_correction")),
        initial_gate=float(checkpoint.get("initial_gate", settings.get("initial_gate", 0.018))),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if checkpoint.get("geometry_frozen", settings.get("freeze_geometry", False)):
        model.freeze_geometry()
    rows = run(model, loader, args.device, float(checkpoint["radial_weight_beta"]))
    errors = np.asarray([row["angular_error_deg"] for row in rows])
    geometry_errors = np.asarray([row["geometry_angular_error_deg"] for row in rows])
    image_errors = np.asarray([row["image_angular_error_deg"] for row in rows])
    summary = {
        "samples": len(rows),
        "mean_angular_error_deg": float(errors.mean()), "median_angular_error_deg": float(np.median(errors)),
        "acc10_pct": float((errors <= 10).mean() * 100),
        "geometry_mean_angular_error_deg": float(geometry_errors.mean()),
        "image_mean_angular_error_deg": float(image_errors.mean()),
        "fused_worse_than_geometry_pct": float((errors > geometry_errors).mean() * 100),
    }
    out_dir = args.out or args.checkpoint.parent / f"inference_{args.split_name}"; out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"checkpoint": str(args.checkpoint.resolve()), "split_name": args.split_name, "source_root": str(args.source_root.resolve()),
                "normal_convention": checkpoint["normal_convention"], "input_schema": schema,
                "obb_detections": str(args.obb_detections.resolve()) if args.obb_detections else None,
                "geometry_sampling_seed": int(sampling_seed)}
    (out_dir / "report.json").write_text(json.dumps({"metadata": metadata, "summary": summary, "predictions": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["file", "pred_nx", "pred_ny", "pred_nz", "target_nx", "target_ny", "target_nz", "geometry_angular_error_deg", "image_angular_error_deg", "angular_error_deg"])
        for row in rows:
            writer.writerow([row["file"], *row["pred_normal"], *row["target_normal"], row["geometry_angular_error_deg"], row["image_angular_error_deg"], row["angular_error_deg"]])
    print(f"result split={args.split_name} mean={summary['mean_angular_error_deg']:.3f}deg median={summary['median_angular_error_deg']:.3f}deg "
          f"<=10deg={summary['acc10_pct']:.1f}% report={out_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
