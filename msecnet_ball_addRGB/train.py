#!/usr/bin/env python3
"""Train a late-fusion RGB plus center-ball geometry normal regressor."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MROOT = PROJECT_ROOT / "msecnet_best" / "MSECNet"
sys.path.insert(0, str(MROOT / "model"))
sys.path.insert(0, str(MROOT / "scripts"))
from architectures import MSECNet  # noqa: E402
from util import config  # noqa: E402

try:  # Support both ``python -m`` and direct script execution from the project root.
    from .data import BallRGBNormalDataset, collate_ball_rgb, source_images_from_index  # noqa: E402
    from .model import PointRGBFusionNormalNet, normalized, radial_weighted_pool  # noqa: E402
except ImportError:
    from data import BallRGBNormalDataset, collate_ball_rgb, source_images_from_index  # noqa: E402
    from model import PointRGBFusionNormalNet, normalized, radial_weighted_pool  # noqa: E402


EPS = 1e-6


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int) -> None:
    value = torch.initial_seed() % (2 ** 32)
    random.seed(value); np.random.seed(value)


def split_indices(files, split_path: Path):
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train, val = set(split.get("train", [])), set(split.get("val", []))
    if not train or not val or train & val:
        raise ValueError("split must have non-empty, disjoint train and val lists")
    names = [str(item) for item in files]
    unknown = (train | val) - set(names)
    if unknown:
        raise ValueError(f"split references {len(unknown)} unknown files")
    train_index = np.array([index for index, name in enumerate(names) if name in train], dtype=np.int64)
    val_index = np.array([index for index, name in enumerate(names) if name in val], dtype=np.int64)
    return train_index, val_index


def cosine_error(vector: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1 - (normalized(vector) * target).sum(1).clamp(-1, 1)


def normal_objective(output, target, counts, radial, beta, point_weight, geometry_weight, image_weight):
    point_directions = F.normalize(output["point_vectors"], dim=1, eps=EPS)
    target_per_point = torch.repeat_interleave(target, counts.to(target.device), dim=0)
    point_error = 1 - (point_directions * target_per_point).sum(1).clamp(-1, 1)
    point_loss = radial_weighted_pool(point_error[:, None], counts, radial, beta).squeeze(1)
    geometry_loss = cosine_error(output["geometry_vector"], target)
    image_loss = cosine_error(output["image_vector"], target)
    fused_loss = cosine_error(output["fused_vector"], target)
    total = (
        point_weight * point_loss + geometry_weight * geometry_loss + image_weight * image_loss
        + (1 - point_weight - geometry_weight - image_weight) * fused_loss
    )
    return total, {"point": point_loss, "geometry": geometry_loss, "image": image_loss, "fused": fused_loss}


@torch.no_grad()
def evaluate(model, loader, device, beta, point_weight, geometry_weight, image_weight):
    model.eval()
    losses, fused_errors, geometry_errors, image_errors = [], [], [], []
    for coord, radial, offset, image, target, _, _, counts in loader:
        output = model(coord.to(device), radial.to(device), offset.to(device), image.to(device), counts, beta)
        target = target.to(device)
        total, _ = normal_objective(output, target, counts, radial.to(device), beta, point_weight, geometry_weight, image_weight)
        losses.append(total.cpu().numpy())
        for values, vector in ((fused_errors, output["fused_vector"]), (geometry_errors, output["geometry_vector"]), (image_errors, output["image_vector"])):
            cosine = (normalized(vector) * target).sum(1).clamp(-1, 1)
            values.append(torch.rad2deg(torch.arccos(cosine)).cpu().numpy())
    fused = np.concatenate(fused_errors); geometry = np.concatenate(geometry_errors); image = np.concatenate(image_errors)
    return {
        "loss": float(np.concatenate(losses).mean()),
        "mean_err": float(fused.mean()), "median_err": float(np.median(fused)), "acc10": float((fused <= 10).mean() * 100),
        "geometry_mean_err": float(geometry.mean()), "image_mean_err": float(image.mean()),
    }


def build_model(image_pretrained: bool = False):
    cfg = config.load_cfg_from_cfg_file(str(HERE / "config" / "msecnet_ball.yaml"))
    cfg.num_classes = 3
    geometry = MSECNet(cfg)
    geometry_dim = geometry.classifier[0].in_features
    return PointRGBFusionNormalNet(geometry, geometry_dim=geometry_dim, pretrained_image=image_pretrained)


def checkpoint_payload(model, args, step, metrics):
    return {
        "model": model.state_dict(), "step": step, "metrics": metrics, "args": vars(args),
        "normal_convention": "oriented_toward_camera",
        "input_schema": {
            "point_coord": "(xyz - center_3d) / ball_radius_m", "point_feature": "radial_distance",
            "rgb": "separate projected source-image crop", "fusion": "late global feature fusion",
        },
        "ball_radius_m": args.ball_radius_m, "radial_weight_beta": args.radial_weight_beta,
    }


def save_checkpoint(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description="Late RGB/point-cloud fusion training for center-ball normals")
    parser.add_argument("labels", type=Path); parser.add_argument("pcd_dir", type=Path)
    parser.add_argument("--source-root", type=Path, required=True, help="existing source dataset containing index.jsonl and images")
    parser.add_argument("--centers", type=Path, required=True); parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "out" / "rgb_fusion_v1")
    parser.add_argument("--steps", type=int, default=30000); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-points", type=int, default=1024); parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=160); parser.add_argument("--image-crop-scale", type=float, default=2.5)
    parser.add_argument("--image-pretrained", action="store_true", help="use torchvision ImageNet ResNet-18 weights (may download once)")
    parser.add_argument("--rgb-dropout", type=float, default=0.20)
    parser.add_argument("--ball-radius-m", type=float, default=0.08); parser.add_argument("--radial-weight-beta", type=float, default=2.0)
    parser.add_argument("--point-loss-weight", type=float, default=0.15); parser.add_argument("--geometry-loss-weight", type=float, default=0.15)
    parser.add_argument("--image-loss-weight", type=float, default=0.15)
    parser.add_argument("--aug-deg", type=float, default=0.0, help="off by default because arbitrary 3D rotations do not transform the RGB view")
    parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260731); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the MSECNet pointops implementation")
    if args.steps < 1 or args.batch_size < 1 or args.max_points < 0 or args.workers < 0 or args.val_every < 1:
        raise ValueError("invalid training count")
    if args.ball_radius_m <= 0 or args.image_size < 32 or args.image_crop_scale <= 0 or not 0 <= args.rgb_dropout < 1:
        raise ValueError("invalid image or radius setting")
    weights_sum = args.point_loss_weight + args.geometry_loss_weight + args.image_loss_weight
    if any(value < 0 for value in (args.point_loss_weight, args.geometry_loss_weight, args.image_loss_weight)) or weights_sum >= 1:
        raise ValueError("auxiliary loss weights must be non-negative and sum to less than one")
    for path in (args.labels, args.pcd_dir, args.source_root, args.centers, args.split):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"output directory is non-empty: {args.out}")
    if args.aug_deg:
        print("warning: --aug-deg rotates geometry and targets but not the RGB image; keep it at 0 for paired fusion", flush=True)
    seed_everything(args.seed); args.out.mkdir(parents=True, exist_ok=True)
    labels = np.load(args.labels); train_index, val_index = split_indices(labels["files"], args.split)
    files, normals = labels["files"], labels["normal"]
    anchors = json.loads(args.centers.read_text(encoding="utf-8")); source_images = source_images_from_index(args.source_root)
    train_dataset = BallRGBNormalDataset(files[train_index], normals[train_index], args.pcd_dir, anchors, source_images,
                                         args.ball_radius_m, args.max_points, args.image_size, args.image_crop_scale, True,
                                         aug_deg=args.aug_deg, seed=args.seed)
    val_dataset = BallRGBNormalDataset(files[val_index], normals[val_index], args.pcd_dir, anchors, source_images,
                                       args.ball_radius_m, args.max_points, args.image_size, args.image_crop_scale, False,
                                       seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0, worker_init_fn=worker_init,
                              generator=generator, collate_fn=collate_ball_rgb)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0, worker_init_fn=worker_init, collate_fn=collate_ball_rgb)
    model = build_model(args.image_pretrained).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    (args.out / "run.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"RGB fusion: train={len(train_dataset)} val={len(val_dataset)} point_max={args.max_points}", flush=True)
    best = float("inf"); iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            coord, radial, offset, image, target, sample_weight, _, counts = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            coord, radial, offset, image, target, sample_weight, _, counts = next(iterator)
        coord, radial, offset, image, target, sample_weight = (
            coord.to(args.device, non_blocking=True), radial.to(args.device, non_blocking=True), offset.to(args.device, non_blocking=True),
            image.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True), sample_weight.to(args.device, non_blocking=True),
        )
        model.train(); output = model(coord, radial, offset, image, counts, args.radial_weight_beta, args.rgb_dropout)
        loss_per_sample, _ = normal_objective(output, target, counts, radial, args.radial_weight_beta,
                                               args.point_loss_weight, args.geometry_loss_weight, args.image_loss_weight)
        loss = (loss_per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(EPS)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        if step % 100 == 0:
            print(f"step={step}/{args.steps} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            metrics = evaluate(model, val_loader, args.device, args.radial_weight_beta, args.point_loss_weight,
                               args.geometry_loss_weight, args.image_loss_weight)
            payload = checkpoint_payload(model, args, step, metrics); save_checkpoint(payload, args.out / "last.pt")
            print(f"val step={step} fused={metrics['mean_err']:.2f}deg geom={metrics['geometry_mean_err']:.2f}deg "
                  f"rgb={metrics['image_mean_err']:.2f}deg <=10={metrics['acc10']:.1f}%", flush=True)
            if metrics["mean_err"] < best:
                best = metrics["mean_err"]; save_checkpoint(payload, args.out / "best.pt")
                print(f"new best={best:.2f}deg", flush=True)
    print(f"done: best fused validation error={best:.2f}deg", flush=True)


if __name__ == "__main__":
    main()
