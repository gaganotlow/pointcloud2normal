#!/usr/bin/env python3
"""Train a PointNet++ to regress one camera-oriented normal per 8 cm ball."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:  # Package import for tests; script import for the documented commands.
    from .data import (
        ACTIVE_SOURCE_DATASET, BallNormalDataset, collate_fixed_points, load_split_indices,
        oriented_angular_error_deg, require_active_ball_dataset, resolve_num_points,
        seed_everything, seed_worker,
    )
    from .model import build_model
except ImportError:  # pragma: no cover - exercised when invoked as a file
    from data import (
        ACTIVE_SOURCE_DATASET, BallNormalDataset, collate_fixed_points, load_split_indices,
        oriented_angular_error_deg, require_active_ball_dataset, resolve_num_points,
        seed_everything, seed_worker,
    )
    from model import build_model


HERE = Path(__file__).resolve().parent
METRIC_COLUMNS = ("step", "train_loss", "lr", "val_loss", "val_mean_ang_err", "val_median_ang_err", "val_acc10_pct")


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class ModelEMA:
    """Exponential moving average of parameters and normalization buffers."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        current = model.state_dict()
        averaged = self.model.state_dict()
        for name, value in averaged.items():
            source = current[name].detach()
            if torch.is_floating_point(value):
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)


def normal_loss(prediction, target):
    """Directed cosine loss: reversing a camera-oriented normal is an error."""
    prediction = F.normalize(prediction, dim=1, eps=1e-6)
    target = F.normalize(target, dim=1, eps=1e-6)
    return 1 - (prediction * target).sum(1).clamp(-1, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses, errors = [], []
    for xyz, radial, target, _ in loader:
        xyz, radial, target = xyz.to(device), radial.to(device), target.to(device)
        prediction = model.predict_normal(xyz, radial)
        losses.append(normal_loss(prediction, target).cpu().numpy())
        errors.append(oriented_angular_error_deg(prediction, target).cpu().numpy())
    losses, errors = np.concatenate(losses), np.concatenate(errors)
    return {
        "loss": float(losses.mean()),
        "mean_err": float(errors.mean()),
        "median_err": float(np.median(errors)),
        "acc10": float((errors <= 10).mean() * 100),
    }


class MetricsLogger:
    def __init__(self, output_dir):
        self.path = Path(output_dir) / "metrics.csv"
        with open(self.path, "w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=METRIC_COLUMNS).writeheader()

    def write(self, **values):
        row = {column: "" for column in METRIC_COLUMNS}
        row.update(values)
        with open(self.path, "a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=METRIC_COLUMNS).writerow(row)


def checkpoint_payload(model, step, metrics, args):
    return {
        "model": model.state_dict(),
        "model_args": model.model_args,
        "architecture": "pointnet2_global_normal_v1",
        "step": int(step),
        "mean_err": metrics["mean_err"],
        "median_err": metrics["median_err"],
        "p10": metrics["acc10"],
        "val_loss": metrics["loss"],
        "normal_convention": "oriented_toward_camera",
        "aggregation": "single_global_pointnet2_head",
        "pooling": "PointNet++ hierarchical max pooling",
        "ball_radius_m": float(args.ball_radius_m),
        "num_points": int(args.num_points),
        "radial_feature": "r = clamp(norm((point - center) / ball_radius_m), 0, 1)",
        "seed": int(args.seed),
        "source_dataset": ACTIVE_SOURCE_DATASET,
        "ema_decay": float(args.ema_decay),
        "grad_clip": float(args.grad_clip),
        "hard_threshold": float(args.hard_threshold),
        "hard_weight": float(args.hard_weight),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
    }


def write_run_metadata(output_dir, args, train_count, val_count):
    payload = {
        "architecture": "pointnet2_global_normal_v1",
        "normal_convention": "oriented_toward_camera",
        "loss": "1 - dot(normalize(prediction), target)",
        "ball_radius_m": args.ball_radius_m,
        "num_points": args.num_points,
        "jitter_std": args.jitter_std,
        "ema_decay": args.ema_decay,
        "hard_report": str(args.hard_report) if args.hard_report else None,
        "hard_threshold": args.hard_threshold,
        "hard_weight": args.hard_weight,
        "overfit_report": str(args.overfit_report) if args.overfit_report else None,
        "overfit_count": args.overfit_count,
        "train_samples": train_count,
        "val_samples": val_count,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (Path(output_dir) / "run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train PointNet++ on center-anchored ball normals")
    parser.add_argument("labels", type=Path)
    parser.add_argument("pcd_dir", type=Path)
    parser.add_argument("--centers", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "out" / "pointnet2_ball_r08_oriented")
    parser.add_argument("--steps", type=int, default=70000)
    parser.add_argument("--batch-size", "--bs", dest="batch_size", type=int, default=24)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--npoints", type=int, default=None, help="deprecated alias for --num-points")
    parser.add_argument("--ball-radius-m", type=float, default=0.08)
    parser.add_argument("--aug-deg", type=float, default=45.0)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="peak learning rate; 3e-4 is the stable v2 default")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--jitter-std", type=float, default=0.01,
                        help="normalized-coordinate Gaussian jitter; 0.005 is a useful low-noise ablation")
    parser.add_argument("--ema-decay", type=float, default=0.995,
                        help="EMA decay used for validation and best.pt; 0 disables EMA")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="global gradient norm clip; 0 disables clipping")
    parser.add_argument("--hard-report", type=Path, default=None,
                        help="optional prior train report.json for static hard-example weighting")
    parser.add_argument("--hard-threshold", type=float, default=10.0,
                        help="report angle above which --hard-weight is applied")
    parser.add_argument("--hard-weight", type=float, default=4.0,
                        help="weight for hard-report samples above threshold")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="optional PointNet++ checkpoint for low-learning-rate fine-tuning")
    parser.add_argument("--overfit-report", type=Path, default=None,
                        help="select the worst train samples from this inference report and overfit them")
    parser.add_argument("--overfit-count", type=int, default=32,
                        help="number of worst train samples for --overfit-report")
    parser.add_argument("--sa1-points", type=int, default=256)
    parser.add_argument("--sa2-points", type=int, default=64)
    parser.add_argument("--sa1-k", type=int, default=32)
    parser.add_argument("--sa2-k", type=int, default=32)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    args.num_points = resolve_num_points(args.num_points, args.npoints)
    if min(args.steps, args.batch_size, args.val_every, args.sa1_points, args.sa2_points, args.sa1_k, args.sa2_k) < 1:
        raise ValueError("steps, batch size, PointNet++ sizes, and validation interval must be positive")
    if args.ball_radius_m <= 0 or args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        raise ValueError("invalid radius, learning rate, weight decay, or dropout")
    if args.jitter_std < 0 or not 0 <= args.ema_decay < 1 or args.grad_clip < 0:
        raise ValueError("jitter, EMA decay, and gradient clipping must be non-negative; EMA decay must be < 1")
    if args.hard_threshold < 0 or args.hard_weight < 1 or not np.isfinite(args.hard_threshold):
        raise ValueError("hard threshold must be non-negative and hard weight must be >= 1")
    if args.hard_report and not args.hard_report.is_file():
        raise FileNotFoundError(args.hard_report)
    if args.overfit_report and not args.overfit_report.is_file():
        raise FileNotFoundError(args.overfit_report)
    if args.overfit_count < 1:
        raise ValueError("--overfit-count must be positive")
    if args.init_checkpoint and not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)
    if args.workers < 0 or args.val_workers < 0 or args.early_stop_patience < 0:
        raise ValueError("worker counts and early-stop patience must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable; pass --device cpu explicitly")
    for path in (args.labels, args.pcd_dir, args.centers, args.split):
        if not path.exists():
            raise FileNotFoundError(path)
    require_active_ball_dataset(args.labels, args.pcd_dir, args.centers, args.split)

    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    labels = np.load(args.labels, allow_pickle=False)
    train_indices, val_indices = load_split_indices(labels["files"], args.split)
    files, normals = labels["files"], labels["normal"]
    if args.overfit_report:
        report = json.loads(args.overfit_report.read_text(encoding="utf-8"))
        rows = report.get("predictions", [])
        errors = {
            str(row["file"]): float(row.get("angular_error_deg", row.get("axis_error_deg", 0.0)))
            for row in rows if isinstance(row, dict) and "file" in row
        }
        train_names = {str(files[index]) for index in train_indices}
        ranked = sorted(
            ((error, name) for name, error in errors.items() if name in train_names),
            key=lambda item: (-item[0], item[1]),
        )
        if len(ranked) < args.overfit_count:
            raise ValueError(
                f"--overfit-report contains only {len(ranked)} train samples; "
                f"cannot select {args.overfit_count}"
            )
        selected_names = {name for _, name in ranked[:args.overfit_count]}
        selected_indices = np.asarray(
            [index for index in train_indices if str(files[index]) in selected_names], dtype=np.int64
        )
        train_indices = selected_indices
        val_indices = selected_indices.copy()
        # A genuine capacity/label diagnostic must not move the input between
        # epochs or average an EMA model that never sees the same exact batch.
        args.aug_deg = 0.0
        args.jitter_std = 0.0
        args.ema_decay = 0.0
        args.dropout = 0.0
        args.hard_report = None
        print(
            f"overfit diagnostic: selected {len(selected_indices)} worst train samples "
            f"from {args.overfit_report}; fixed points, no augmentation, no EMA/dropout",
            flush=True,
        )
    with open(args.centers, encoding="utf-8") as stream:
        anchors = json.load(stream)
    train_weights = None
    if args.hard_report:
        report = json.loads(args.hard_report.read_text(encoding="utf-8"))
        report_rows = report.get("predictions", [])
        hard_errors = {
            str(row["file"]): float(row.get("angular_error_deg", row.get("axis_error_deg", 0.0)))
            for row in report_rows if isinstance(row, dict) and "file" in row
        }
        missing_hard_rows = [str(file_name) for file_name in files[train_indices] if str(file_name) not in hard_errors]
        if missing_hard_rows:
            raise ValueError(
                f"--hard-report lacks {len(missing_hard_rows)} current training samples; "
                "generate it with pointnet2_ball/infer.py --split-name train"
            )
        train_weights = np.asarray([
            args.hard_weight if hard_errors.get(str(file_name), 0.0) > args.hard_threshold else 1.0
            for file_name in files[train_indices]
        ], dtype=np.float32)
        print(
            f"hard-example weighting: {int((train_weights > 1).sum())}/{len(train_weights)} samples "
            f"at {args.hard_weight:g}x (threshold>{args.hard_threshold:g}deg)", flush=True,
        )
    train_set = BallNormalDataset(
        files[train_indices], normals[train_indices], args.pcd_dir, anchors, args.ball_radius_m,
        args.num_points, train=not bool(args.overfit_report), aug_deg=args.aug_deg, weights=train_weights, seed=args.seed,
        jitter_std=args.jitter_std,
    )
    val_set = BallNormalDataset(
        files[val_indices], normals[val_indices], args.pcd_dir, anchors, args.ball_radius_m,
        args.num_points, train=False, seed=args.seed,
    )
    if len(train_set) < args.batch_size:
        raise ValueError(f"training split has {len(train_set)} samples but batch size is {args.batch_size}")
    generator = torch.Generator().manual_seed(args.seed)
    pin_memory = args.device.startswith("cuda")
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
        persistent_workers=args.workers > 0, pin_memory=pin_memory, worker_init_fn=seed_worker,
        generator=generator, collate_fn=collate_fixed_points,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.val_workers,
        persistent_workers=args.val_workers > 0, pin_memory=pin_memory, worker_init_fn=seed_worker,
        collate_fn=collate_fixed_points,
    )
    requested_model_args = {
        "sa1_points": args.sa1_points, "sa2_points": args.sa2_points,
        "sa1_k": args.sa1_k, "sa2_k": args.sa2_k, "dropout": args.dropout,
    }
    initial_checkpoint = None
    if args.init_checkpoint:
        initial_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial_checkpoint.get("architecture") != "pointnet2_global_normal_v1" or "model" not in initial_checkpoint:
            raise ValueError("--init-checkpoint is not a PointNet++ center-ball checkpoint")
        if initial_checkpoint.get("source_dataset") != ACTIVE_SOURCE_DATASET:
            raise ValueError("--init-checkpoint was not trained on the active 20260803 dataset")
        requested_model_args = initial_checkpoint.get("model_args", requested_model_args)
    model = build_model(requested_model_args).to(args.device)
    if initial_checkpoint:
        model.load_state_dict(initial_checkpoint["model"], strict=True)
        print(
            f"loaded initialization checkpoint {args.init_checkpoint} "
            f"(step={initial_checkpoint.get('step', 'unknown')})", flush=True,
        )
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    logger = MetricsLogger(args.out)
    write_run_metadata(args.out, args, len(train_set), len(val_set))
    print(f"PointNet++: train={len(train_set)} val={len(val_set)} points={args.num_points} device={args.device}", flush=True)

    best_error, stale_checks = float("inf"), 0
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            xyz, radial, target, sample_weights = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xyz, radial, target, sample_weights = next(iterator)
        xyz, radial, target, sample_weights = (
            xyz.to(args.device, non_blocking=True), radial.to(args.device, non_blocking=True),
            target.to(args.device, non_blocking=True), sample_weights.to(args.device, non_blocking=True),
        )
        model.train()
        prediction = model(xyz, radial)
        per_sample = normal_loss(prediction, target)
        loss = (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        scheduler.step()
        if step % 100 == 0:
            logger.write(step=step, train_loss=float(loss.detach().cpu()), lr=scheduler.get_last_lr()[0])
            print(f"step {step}/{args.steps} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
        if step % args.val_every != 0 and step != args.steps:
            continue
        eval_model = ema.model if ema is not None else model
        metrics = evaluate(eval_model, val_loader, args.device)
        logger.write(
            step=step, val_loss=metrics["loss"], val_mean_ang_err=metrics["mean_err"],
            val_median_ang_err=metrics["median_err"], val_acc10_pct=metrics["acc10"],
        )
        payload = checkpoint_payload(eval_model, step, metrics, args)
        atomic_torch_save(payload, args.out / "last.pt")
        print(
            f"[VAL {step}] mean={metrics['mean_err']:.2f}deg median={metrics['median_err']:.2f}deg "
            f"<=10deg={metrics['acc10']:.1f}% loss={metrics['loss']:.4f}", flush=True,
        )
        if metrics["mean_err"] < best_error:
            best_error, stale_checks = metrics["mean_err"], 0
            atomic_torch_save(payload, args.out / "best.pt")
            print(f"  -> new best {best_error:.2f}deg", flush=True)
        else:
            stale_checks += 1
        if args.early_stop_patience and stale_checks >= args.early_stop_patience:
            print(f"early stop after {stale_checks} validation checks without improvement", flush=True)
            break
    print(f"done: best val mean angular error={best_error:.3f}deg", flush=True)


if __name__ == "__main__":
    main()
