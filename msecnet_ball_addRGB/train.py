#!/usr/bin/env python3
"""Train a late-fusion RGB plus center-ball geometry normal regressor."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
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
    from .data import BallRGBNormalDataset, collate_ball_rgb, load_obb_detections, source_images_from_index  # noqa: E402
    from .model import PointRGBFusionNormalNet, normalized, radial_weighted_pool  # noqa: E402
except ImportError:
    from data import BallRGBNormalDataset, collate_ball_rgb, load_obb_detections, source_images_from_index  # noqa: E402
    from model import PointRGBFusionNormalNet, normalized, radial_weighted_pool  # noqa: E402


EPS = 1e-6
ACTIVE_SOURCE_DATASET = "fuelcap_pass_20260803_10847"
METRIC_COLUMNS = (
    "step", "train_loss", "lr", "val_loss", "val_mean_ang_err",
    "val_median_ang_err", "val_acc10_pct", "val_geometry_mean_ang_err",
    "val_image_mean_ang_err", "val_fusion_gate_mean",
)


def require_active_rgb_dataset(labels_path: Path, pcd_dir: Path, centers_path: Path, split_path: Path, source_root: Path) -> None:
    """Require one 10847 center-ball dataset and its matching raw RGB source."""
    dataset_dir = labels_path.resolve().parent
    expected_paths = (
        dataset_dir / "labels_manual3d.npz",
        dataset_dir / "clouds",
        dataset_dir / "anchors_manual3d.json",
        dataset_dir / "split_by_generalization_group.json",
    )
    if tuple(path.resolve() for path in (labels_path, pcd_dir, centers_path, split_path)) != tuple(
        path.resolve() for path in expected_paths
    ):
        raise ValueError("labels, clouds, centers, and split must belong to one prepared center-ball dataset")
    metadata_path = dataset_dir / "dataset.json"
    if not metadata_path.is_file():
        raise ValueError(f"{dataset_dir} has no dataset.json provenance; prepare it from {ACTIVE_SOURCE_DATASET}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_source_root = (PROJECT_ROOT / "data" / ACTIVE_SOURCE_DATASET).resolve()
    if (
        metadata.get("kind") != "center_ball"
        or metadata.get("source_dataset") != ACTIVE_SOURCE_DATASET
        or source_root.resolve() != expected_source_root
    ):
        raise ValueError(f"RGB training only accepts data derived from {expected_source_root}")


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


def normal_objective(output, target, counts, radial, beta, point_weight, geometry_weight, image_weight,
                     gate_penalty: float = 0.0):
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
    gate = output["fusion_gate"].squeeze(1)
    total = total + gate_penalty * gate.square()
    return total, {"point": point_loss, "geometry": geometry_loss, "image": image_loss, "fused": fused_loss, "gate": gate}


@torch.no_grad()
def evaluate(model, loader, device, beta, point_weight, geometry_weight, image_weight, gate_penalty: float = 0.0):
    model.eval()
    losses, fused_errors, geometry_errors, image_errors, fusion_gates = [], [], [], [], []
    for coord, radial, offset, image, target, _, _, counts, point_uv in loader:
        output = model(coord.to(device), radial.to(device), offset.to(device), image.to(device), counts, beta,
                       point_uv=point_uv.to(device))
        target = target.to(device)
        total, _ = normal_objective(
            output, target, counts, radial.to(device), beta, point_weight, geometry_weight, image_weight, gate_penalty,
        )
        losses.append(total.cpu().numpy())
        fusion_gates.append(output["fusion_gate"].squeeze(1).cpu().numpy())
        for values, vector in ((fused_errors, output["fused_vector"]), (geometry_errors, output["geometry_vector"]), (image_errors, output["image_vector"])):
            cosine = (normalized(vector) * target).sum(1).clamp(-1, 1)
            values.append(torch.rad2deg(torch.arccos(cosine)).cpu().numpy())
    fused = np.concatenate(fused_errors); geometry = np.concatenate(geometry_errors); image = np.concatenate(image_errors)
    gates = np.concatenate(fusion_gates)
    return {
        "loss": float(np.concatenate(losses).mean()),
        "mean_err": float(fused.mean()), "median_err": float(np.median(fused)), "acc10": float((fused <= 10).mean() * 100),
        "geometry_mean_err": float(geometry.mean()), "image_mean_err": float(image.mean()),
        "fusion_gate_mean": float(gates.mean()),
    }


def build_model(image_pretrained: bool = True, image_backbone: str = "dino_vits14", dino_unfreeze_blocks: int = 2,
                fusion_mode: str = "gated_residual", geometry_mode: str = "feature_head",
                max_rgb_correction: float | None = None, initial_gate: float = 0.018):
    cfg = config.load_cfg_from_cfg_file(str(HERE / "config" / "msecnet_ball.yaml"))
    cfg.num_classes = 3
    geometry = MSECNet(cfg)
    geometry_dim = geometry.classifier[0].in_features
    return PointRGBFusionNormalNet(
        geometry, geometry_dim=geometry_dim, pretrained_image=image_pretrained,
        image_backbone=image_backbone, dino_unfreeze_blocks=dino_unfreeze_blocks, fusion_mode=fusion_mode,
        geometry_mode=geometry_mode, max_rgb_correction=max_rgb_correction, initial_gate=initial_gate,
    )


def load_geometry_checkpoint(model, checkpoint_path: Path, args, checkpoint: dict | None = None) -> dict:
    """Load a point-only MSECNet checkpoint into the preserved geometry branch."""
    if model.geometry_mode != "pretrained_point":
        raise ValueError("--geometry-checkpoint requires --geometry-mode pretrained_point")
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"{checkpoint_path} is missing model weights")
    expected = {
        "normal_convention": "oriented_toward_camera",
        "ball_radius_m": args.ball_radius_m,
        "radial_weight_beta": args.radial_weight_beta,
        "source_dataset": ACTIVE_SOURCE_DATASET,
    }
    for key, value in expected.items():
        actual = checkpoint.get(key)
        if key == "source_dataset" and actual != value:
            raise ValueError(f"geometry checkpoint {key}={actual!r} is incompatible with requested {value!r}")
        if actual is None:
            continue
        if isinstance(value, float):
            matches = np.isclose(float(actual), value)
        else:
            matches = actual == value
        if not matches:
            raise ValueError(f"geometry checkpoint {key}={actual!r} is incompatible with requested {value!r}")
    model.geometry.load_state_dict(checkpoint["model"], strict=True)
    print(f"loaded frozen-point geometry initialization from {checkpoint_path} (step={checkpoint.get('step', 'unknown')})", flush=True)
    return checkpoint


def checkpoint_payload(model, args, step, metrics):
    return {
        "model": model.state_dict(), "step": step, "metrics": metrics, "args": vars(args),
        "normal_convention": "oriented_toward_camera",
        "source_dataset": ACTIVE_SOURCE_DATASET,
        "input_schema": {
            "point_coord": "(xyz - center_3d) / ball_radius_m", "point_feature": "radial_distance",
            "rgb": "camera-oriented detector-OBB source-image crop" if args.obb_detections else "separate projected source-image crop",
            "rgb_crop_source": "detector_obb" if args.obb_detections else "projected_center",
            "fusion": (
                "frozen point baseline + gated residual RGB fusion"
                if args.geometry_mode == "pretrained_point" and args.freeze_geometry and args.fusion_mode == "gated_residual"
                else "point-aligned DINO patch residual RGB fusion" if args.fusion_mode == "point_aligned_residual"
                else "gated residual global feature fusion" if args.fusion_mode == "gated_residual" else "late global feature fusion"
            ),
            "rgb_point_alignment": "projected_ball_points_to_camera_oriented_obb_crop" if args.fusion_mode == "point_aligned_residual" else None,
        },
        "ball_radius_m": args.ball_radius_m, "radial_weight_beta": args.radial_weight_beta,
        "obb_crop_scale": args.obb_crop_scale if args.obb_detections else None,
        "image_backbone": args.image_backbone,
        "dino_unfreeze_blocks": args.dino_unfreeze_blocks,
        "dino_lr": args.dino_lr,
        "fusion_mode": args.fusion_mode,
        "geometry_mode": args.geometry_mode,
        "geometry_checkpoint": str(args.geometry_checkpoint.resolve()) if args.geometry_checkpoint else None,
        "geometry_frozen": args.freeze_geometry,
        "geometry_sampling_seed": args.geometry_sampling_seed,
        "max_rgb_correction": args.max_rgb_correction,
        "gate_penalty": args.gate_penalty,
        "initial_gate": args.initial_gate,
    }


def save_checkpoint(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


class TrainLogger:
    """Persist scalar metrics and render the current training dashboard."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.csv_path = out_dir / "metrics.csv"
        self.history = defaultdict(list)
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=METRIC_COLUMNS).writeheader()

    def log_train(self, step: int, loss: float, lr: float) -> None:
        self._append({"step": step, "train_loss": loss, "lr": lr})

    def log_val(self, step: int, metrics: dict) -> None:
        self._append({
            "step": step,
            "val_loss": metrics["loss"],
            "val_mean_ang_err": metrics["mean_err"],
            "val_median_ang_err": metrics["median_err"],
            "val_acc10_pct": metrics["acc10"],
            "val_geometry_mean_ang_err": metrics["geometry_mean_err"],
            "val_image_mean_ang_err": metrics["image_mean_err"],
            "val_fusion_gate_mean": metrics["fusion_gate_mean"],
        })

    def _append(self, values: dict) -> None:
        row = {name: "" for name in METRIC_COLUMNS}
        row.update(values)
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=METRIC_COLUMNS).writerow(row)
        step = int(values["step"])
        for name, value in values.items():
            if name != "step":
                self.history[name].append(float(value))
                self.history[f"{name}_step"].append(step)

    def plot_dashboard(self, step: int) -> Path | None:
        """Write the latest loss, validation, accuracy, and learning-rate plots."""
        if not HAS_MPL:
            return None
        figure, axes = plt.subplots(2, 2, figsize=(14, 10))
        figure.suptitle(f"RGB Fusion MSECNet Training - step {step}", fontsize=13, fontweight="bold")
        history = self.history

        axis = axes[0, 0]
        self._plot(axis, "train_loss", "Training loss", color="tab:blue")

        axis = axes[0, 1]
        self._plot(axis, "val_mean_ang_err", "Validation mean angle error (deg)", "fused", "tab:orange")
        self._plot(axis, "val_geometry_mean_ang_err", "Validation mean angle error (deg)", "geometry", "tab:blue")
        self._plot(axis, "val_image_mean_ang_err", "Validation mean angle error (deg)", "RGB", "tab:green")
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)

        axis = axes[1, 0]
        self._plot(axis, "val_acc10_pct", "Validation within 10 deg (%)", color="tab:green")
        axis.set_ylim(0, 105)

        axis = axes[1, 1]
        self._plot(axis, "lr", "Learning rate", color="tab:red")
        gate_axis = axis.twinx()
        self._plot(gate_axis, "val_fusion_gate_mean", "Mean RGB gate", "gate", "tab:purple")
        gate_axis.set_ylim(0, 1)
        if gate_axis.get_legend_handles_labels()[0]:
            gate_axis.legend(loc="lower right", fontsize=8)

        for axis in axes.flat:
            axis.set_xlabel("Step")
            axis.grid(True, alpha=0.3)
        figure.tight_layout()
        path = self.out_dir / "dashboard.png"
        figure.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        return path

    def _plot(self, axis, metric: str, ylabel: str, label: str | None = None, color=None) -> None:
        values = self.history.get(metric, [])
        steps = self.history.get(f"{metric}_step", [])
        if values:
            axis.plot(steps, values, "o-" if metric.startswith("val_") else "-", label=label,
                      color=color, markersize=3, linewidth=1.2)
        axis.set_ylabel(ylabel)


def main():
    parser = argparse.ArgumentParser(description="Late RGB/point-cloud fusion training for center-ball normals")
    parser.add_argument("labels", type=Path); parser.add_argument("pcd_dir", type=Path)
    parser.add_argument("--source-root", type=Path, required=True, help="existing source dataset containing index.jsonl and images")
    parser.add_argument("--centers", type=Path, required=True); parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "out" / "rgb_fusion_v1")
    parser.add_argument("--steps", type=int, default=30000); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-points", type=int, default=1024); parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224); parser.add_argument("--image-crop-scale", type=float, default=2.5)
    parser.add_argument("--obb-detections", type=Path, default=None,
                        help="JSON from detect_obb.py; uses camera-oriented detector OBB crops instead of projected-center crops")
    parser.add_argument("--obb-crop-scale", type=float, default=1.4,
                        help="expand detector OBB width and height by this factor before camera-oriented cropping")
    parser.add_argument("--image-backbone", choices=("dino_vits14", "resnet18"), default="dino_vits14")
    parser.add_argument("--dino-unfreeze-blocks", type=int, default=2,
                        help="DINOv2 final Transformer blocks to fine-tune; 0 freezes the pretrained backbone")
    parser.add_argument("--dino-lr", type=float, default=1e-5,
                        help="peak learning rate for unfrozen DINOv2 blocks")
    parser.add_argument("--fusion-mode", choices=("gated_residual", "point_aligned_residual", "legacy"), default="gated_residual",
                        help="point_aligned_residual samples DINO patch features at projected ball points; gated_residual is global")
    parser.add_argument("--geometry-mode", choices=("feature_head", "pretrained_point"), default="feature_head",
                        help="pretrained_point retains the original MSECNet classifier for an initialized point-only baseline")
    parser.add_argument("--geometry-checkpoint", type=Path, default=None,
                        help="point-only MSECNet checkpoint to load into --geometry-mode pretrained_point")
    parser.add_argument("--freeze-geometry", action="store_true",
                        help="freeze the initialized point-only MSECNet; recommended for RGB residual training")
    parser.add_argument("--max-rgb-correction", type=float, default=0.05,
                        help="maximum norm of the gated RGB residual before normalizing the final vector")
    parser.add_argument("--gate-penalty", type=float, default=0.0,
                        help="per-sample penalty multiplier for squared RGB gate values")
    parser.add_argument("--initial-gate", type=float, default=0.10,
                        help="initial RGB residual gate; 0.10 gives the residual useful learning gradients while preserving geometry")
    parser.add_argument("--image-pretrained", action="store_true",
                        help="use torchvision ImageNet weights when --image-backbone=resnet18")
    parser.add_argument("--rgb-dropout", type=float, default=0.20)
    parser.add_argument("--ball-radius-m", type=float, default=0.08); parser.add_argument("--radial-weight-beta", type=float, default=2.0)
    parser.add_argument("--point-loss-weight", type=float, default=0.15); parser.add_argument("--geometry-loss-weight", type=float, default=0.15)
    parser.add_argument("--image-loss-weight", type=float, default=0.15)
    parser.add_argument("--aug-deg", type=float, default=0.0, help="off by default because arbitrary 3D rotations do not transform the RGB view")
    parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260731); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the MSECNet pointops implementation")
    if (args.steps < 1 or args.batch_size < 1 or args.max_points < 0 or args.workers < 0
            or args.log_every < 1 or args.val_every < 1):
        raise ValueError("invalid training count")
    if (args.ball_radius_m <= 0 or args.image_size < 32 or args.image_crop_scale <= 0 or args.obb_crop_scale <= 0
            or not 0 <= args.rgb_dropout < 1):
        raise ValueError("invalid image or radius setting")
    if args.image_backbone == "dino_vits14" and args.image_size % 14:
        raise ValueError("--image-size must be divisible by 14 for DINOv2 ViT-S/14")
    if args.dino_unfreeze_blocks < 0 or args.dino_lr <= 0:
        raise ValueError("--dino-unfreeze-blocks must be non-negative and --dino-lr must be positive")
    if args.max_rgb_correction < 0 or args.gate_penalty < 0 or not 0 < args.initial_gate < 1:
        raise ValueError("--max-rgb-correction and --gate-penalty must be non-negative; --initial-gate must be in (0, 1)")
    if args.geometry_checkpoint is not None and not args.geometry_checkpoint.is_file():
        raise FileNotFoundError(args.geometry_checkpoint)
    if args.geometry_mode == "pretrained_point" and args.geometry_checkpoint is None:
        raise ValueError("--geometry-mode pretrained_point requires --geometry-checkpoint")
    if args.freeze_geometry and args.geometry_mode != "pretrained_point":
        raise ValueError("--freeze-geometry requires --geometry-mode pretrained_point")
    if args.fusion_mode == "point_aligned_residual" and args.obb_detections is None:
        raise ValueError("--fusion-mode point_aligned_residual requires --obb-detections for point-to-crop projection")
    if args.fusion_mode == "point_aligned_residual" and args.obb_crop_scale < 3.0:
        raise ValueError("point_aligned_residual requires --obb-crop-scale >= 3.0 so projected ball points remain inside the crop")
    weights_sum = args.point_loss_weight + args.geometry_loss_weight + args.image_loss_weight
    if any(value < 0 for value in (args.point_loss_weight, args.geometry_loss_weight, args.image_loss_weight)) or weights_sum >= 1:
        raise ValueError("auxiliary loss weights must be non-negative and sum to less than one")
    for path in (args.labels, args.pcd_dir, args.source_root, args.centers, args.split):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.obb_detections is not None and not args.obb_detections.is_file():
        raise FileNotFoundError(args.obb_detections)
    require_active_rgb_dataset(args.labels, args.pcd_dir, args.centers, args.split, args.source_root)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"output directory is non-empty: {args.out}")
    if args.aug_deg:
        print("warning: --aug-deg rotates geometry and targets but not the RGB image; keep it at 0 for paired fusion", flush=True)
    geometry_initialization = None
    args.geometry_sampling_seed = args.seed
    if args.geometry_checkpoint is not None:
        geometry_initialization = torch.load(args.geometry_checkpoint, map_location="cpu", weights_only=False)
        args.geometry_sampling_seed = int(geometry_initialization.get("seed", args.seed))
    seed_everything(args.seed); args.out.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(args.out)
    labels = np.load(args.labels); train_index, val_index = split_indices(labels["files"], args.split)
    files, normals = labels["files"], labels["normal"]
    anchors = json.loads(args.centers.read_text(encoding="utf-8")); source_images = source_images_from_index(args.source_root)
    obb_detections = load_obb_detections(args.obb_detections) if args.obb_detections else None
    train_sampling_indices, val_sampling_indices = np.arange(len(train_index)), np.arange(len(val_index))
    if obb_detections is not None:
        def detected(indices, sampling_indices):
            keep = np.asarray([str(files[index]) in obb_detections for index in indices], dtype=bool)
            return indices[keep], sampling_indices[keep]

        original_train, original_val = len(train_index), len(val_index)
        train_index, train_sampling_indices = detected(train_index, train_sampling_indices)
        val_index, val_sampling_indices = detected(val_index, val_sampling_indices)
        print(
            f"OBB RGB crops: train={len(train_index)}/{original_train} val={len(val_index)}/{original_val} "
            f"(excluded {original_train - len(train_index) + original_val - len(val_index)} samples without detections)",
            flush=True,
        )
    if len(train_index) < args.batch_size:
        raise ValueError(f"training split has {len(train_index)} usable samples but batch size is {args.batch_size}")
    if not len(val_index):
        raise ValueError("validation split has no usable samples")
    train_dataset = BallRGBNormalDataset(files[train_index], normals[train_index], args.pcd_dir, anchors, source_images,
                                         args.ball_radius_m, args.max_points, args.image_size, args.image_crop_scale, True,
                                         aug_deg=args.aug_deg, seed=args.seed, obb_detections=obb_detections,
                                         obb_crop_scale=args.obb_crop_scale, sampling_indices=train_sampling_indices,
                                         sampling_seed=args.geometry_sampling_seed)
    val_dataset = BallRGBNormalDataset(files[val_index], normals[val_index], args.pcd_dir, anchors, source_images,
                                       args.ball_radius_m, args.max_points, args.image_size, args.image_crop_scale, False,
                                       seed=args.seed, obb_detections=obb_detections, obb_crop_scale=args.obb_crop_scale,
                                       sampling_indices=val_sampling_indices, sampling_seed=args.geometry_sampling_seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0, worker_init_fn=worker_init,
                              generator=generator, collate_fn=collate_ball_rgb)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0, worker_init_fn=worker_init, collate_fn=collate_ball_rgb)
    model = build_model(
        args.image_pretrained, image_backbone=args.image_backbone,
        dino_unfreeze_blocks=args.dino_unfreeze_blocks, fusion_mode=args.fusion_mode,
        geometry_mode=args.geometry_mode, max_rgb_correction=args.max_rgb_correction, initial_gate=args.initial_gate,
    ).to(args.device)
    if args.geometry_checkpoint is not None:
        load_geometry_checkpoint(model, args.geometry_checkpoint, args, checkpoint=geometry_initialization)
    if args.freeze_geometry:
        model.freeze_geometry()
    dino_parameters = [parameter for name, parameter in model.named_parameters()
                       if name.startswith("rgb.backbone.") and parameter.requires_grad]
    dino_parameter_ids = {id(parameter) for parameter in dino_parameters}
    main_parameters = [parameter for parameter in model.parameters()
                       if parameter.requires_grad and id(parameter) not in dino_parameter_ids]
    parameter_groups = [{"params": main_parameters, "lr": args.lr}]
    max_lrs = [args.lr]
    if dino_parameters:
        parameter_groups.append({"params": dino_parameters, "lr": args.dino_lr})
        max_lrs.append(args.dino_lr)
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lrs, total_steps=args.steps, pct_start=0.05)
    (args.out / "run.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"RGB fusion: train={len(train_dataset)} val={len(val_dataset)} point_max={args.max_points}", flush=True)
    best = float("inf"); iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            coord, radial, offset, image, target, sample_weight, _, counts, point_uv = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            coord, radial, offset, image, target, sample_weight, _, counts, point_uv = next(iterator)
        coord, radial, offset, image, target, sample_weight, point_uv = (
            coord.to(args.device, non_blocking=True), radial.to(args.device, non_blocking=True), offset.to(args.device, non_blocking=True),
            image.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True), sample_weight.to(args.device, non_blocking=True),
            point_uv.to(args.device, non_blocking=True),
        )
        model.train(); output = model(coord, radial, offset, image, counts, args.radial_weight_beta, args.rgb_dropout, point_uv)
        loss_per_sample, _ = normal_objective(output, target, counts, radial, args.radial_weight_beta,
                                               args.point_loss_weight, args.geometry_loss_weight, args.image_loss_weight,
                                               args.gate_penalty)
        loss = (loss_per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(EPS)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        if step % args.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            logger.log_train(step, loss.item(), lr)
            print(f"step={step}/{args.steps} loss={loss.item():.4f} lr={lr:.2e}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            metrics = evaluate(model, val_loader, args.device, args.radial_weight_beta, args.point_loss_weight,
                               args.geometry_loss_weight, args.image_loss_weight, args.gate_penalty)
            logger.log_val(step, metrics)
            dashboard = logger.plot_dashboard(step)
            payload = checkpoint_payload(model, args, step, metrics); save_checkpoint(payload, args.out / "last.pt")
            print(f"val step={step} fused={metrics['mean_err']:.2f}deg geom={metrics['geometry_mean_err']:.2f}deg "
                  f"rgb={metrics['image_mean_err']:.2f}deg gate={metrics['fusion_gate_mean']:.3f} "
                  f"<=10={metrics['acc10']:.1f}%", flush=True)
            if dashboard is None:
                print("warning: matplotlib is unavailable; metrics.csv was saved but dashboard.png was not created", flush=True)
            if metrics["mean_err"] < best:
                best = metrics["mean_err"]; save_checkpoint(payload, args.out / "best.pt")
                print(f"new best={best:.2f}deg", flush=True)
    print(f"done: best fused validation error={best:.2f}deg", flush=True)


if __name__ == "__main__":
    main()
