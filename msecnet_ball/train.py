#!/usr/bin/env python3
"""Train MSECNet to estimate an oriented normal at a human sphere center.

MSECNet produces a vector per input point. Manual labels are consistently
oriented toward the camera, so training normalizes the point vectors and
optimizes the normalized mean vector that is used at inference time. A smaller
per-point term keeps the individual vectors coherent with that patch target.
Run in the ``point2normal`` Conda environment.
"""
import argparse
import csv
import random
import os
import sys
from collections import defaultdict

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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MROOT = os.path.join(ROOT, "msecnet", "MSECNet")
sys.path.insert(0, os.path.join(MROOT, "model"))     # 'blocks'
sys.path.insert(0, os.path.join(MROOT, "scripts"))   # 'lib' (pointops), 'util'
from util import config                              # noqa: E402
from architectures import MSECNet                    # noqa: E402
from torch.utils.data import Dataset                 # noqa: E402
import json                                          # noqa: E402


NORMAL_EPS = 1e-6
METRIC_COLUMNS = (
    "step", "train_loss", "lr", "val_loss", "val_mean_ang_err",
    "val_median_ang_err", "val_acc10_pct", "val_point_consensus",
)


def rand_rot(rs, max_deg=180.0):
    axis = rs.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
    ang = np.radians(rs.uniform(0, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return R.astype(np.float32)


class BallNormalDS(Dataset):
    """Return a fixed-radius ball in coordinates anchored at its target point."""
    def __init__(self, files, normals, pcd_dir, max_points, train, anchors, ball_radius_m, weights=None, aug_deg=180.0, seed=None):
        self.files, self.normals, self.pcd_dir, self.max_points, self.train = files, normals, pcd_dir, max_points, train
        self.anchors = anchors; self.ball_radius_m = ball_radius_m
        self.weights = weights; self.aug_deg = aug_deg; self.seed = seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(os.path.join(self.pcd_dir, self.files[i]))
        xyz = d["xyz"][d["label"] == 1].astype(np.float32)
        n = self.normals[i].astype(np.float32).copy()
        anchor = self.anchors.get(str(self.files[i]))
        if anchor is None or anchor.get("selection") != "manual_center_ball_patch":
            raise ValueError(f"{self.files[i]} is not a prepared center-ball sample")
        source_radius = float(anchor.get("ball_radius_m", 0))
        if not np.isclose(source_radius, self.ball_radius_m, rtol=0, atol=1e-8):
            raise ValueError(
                f"{self.files[i]} was prepared with ball_radius_m={source_radius}, "
                f"not {self.ball_radius_m}"
            )
        if len(xyz) == 0:
            raise ValueError(f"{self.files[i]} has no prepared ball points")
        # Training workers receive a reproducible NumPy seed. Validation uses a
        # sample-local generator, so every validation pass sees exactly the same points.
        # ``seed=None`` preserves the original legacy inference sampling sequence.
        rs = np.random if self.train else (
            np.random.RandomState(i) if self.seed is None else np.random.default_rng(self.seed + i * 7919)
        )
        if self.max_points > 0 and len(xyz) > self.max_points:
            xyz = xyz[rs.choice(len(xyz), self.max_points, replace=False)]
        xyz = (xyz - np.asarray(anchor["center_3d"], dtype=np.float32)) / self.ball_radius_m
        if self.train:
            R = rand_rot(rs, self.aug_deg); xyz = xyz @ R.T; n = R @ n
            xyz += rs.normal(0, 0.01, xyz.shape).astype(np.float32)
        # The scalar radius tells the model how far each point is from the target
        # point at the coordinate origin.  Clamp preserves the [0, 1] contract
        # after augmentation jitter near the sphere boundary.
        radial_distance = np.minimum(np.linalg.norm(xyz, axis=1, keepdims=True), 1.0)
        n = n / (np.linalg.norm(n) + 1e-9)
        w = float(self.weights[i]) if self.weights is not None else 1.0
        return xyz.astype(np.float32), radial_distance.astype(np.float32), n.astype(np.float32), np.float32(w)


def collate_variable_points(batch):
    """Pack variable-length point clouds for MSECNet's offset-based interface."""
    coords = [torch.from_numpy(item[0]) for item in batch]
    features = [torch.from_numpy(item[1]) for item in batch]
    counts = torch.tensor([len(coord) for coord in coords], dtype=torch.int64)
    return (
        torch.cat(coords, dim=0).float(),
        torch.cat(features, dim=0).float(),
        torch.cumsum(counts, dim=0).int(),
        torch.from_numpy(np.stack([item[2] for item in batch])).float(),
        torch.tensor([item[3] for item in batch]).float(),
        counts,
    )


def to_msecnet(coord, radial_distance, offset):
    """Use normalized point-to-center distance as MSECNet's one input feature."""
    return coord, radial_distance, offset


def radial_weights(radial_distance, beta):
    """Near-center emphasis; at r=1 a point has weight exp(-beta)."""
    return torch.exp(-beta * radial_distance.squeeze(1).square())


def aggregate_point_normals(point_normals, counts, point_weights):
    """Aggregate normalized point directions into one oriented normal per patch.

    ``consensus`` is in [0, 1] for well-formed predictions: 1 means all point
    directions agree, while values near 0 reveal cancellation or disagreement.
    """
    directions = F.normalize(point_normals, dim=1, eps=NORMAL_EPS)
    means = torch.stack([
        (points * weights[:, None]).sum(0) / weights.sum().clamp_min(NORMAL_EPS)
        for points, weights in zip(torch.split(directions, counts.tolist()), torch.split(point_weights, counts.tolist()))
    ])
    return means, F.normalize(means, dim=1, eps=NORMAL_EPS), directions


def oriented_angular_error_deg(prediction, target):
    """Per-sample angular error for camera-oriented unit normals."""
    cosine = (prediction * target).sum(1).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cosine))


def normal_losses(point_normals, target, counts, radial_distance, point_loss_weight, radial_weight_beta):
    """Return weighted patch losses for the deployed oriented aggregation rule."""
    point_weights = radial_weights(radial_distance, radial_weight_beta)
    mean_vectors, patch_normal, directions = aggregate_point_normals(point_normals, counts, point_weights)
    target_per_point = torch.repeat_interleave(target, counts.to(target.device), dim=0)
    point_errors = 1 - (directions * target_per_point).sum(1).clamp(-1, 1)
    point_loss = torch.stack([
        (errors * weights).sum() / weights.sum().clamp_min(NORMAL_EPS)
        for errors, weights in zip(torch.split(point_errors, counts.tolist()), torch.split(point_weights, counts.tolist()))
    ])
    patch_loss = 1 - (patch_normal * target).sum(1).clamp(-1, 1)
    loss = point_loss_weight * point_loss + (1 - point_loss_weight) * patch_loss
    return loss, point_loss, patch_loss, patch_normal, mean_vectors


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch RNGs used by this training process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Give each DataLoader worker a deterministic, distinct RNG stream."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def atomic_torch_save(payload, path):
    """Avoid leaving a partially-written checkpoint after interruption."""
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def checkpoint_payload(model, step, metrics, args):
    """Build the checkpoint format shared by initial and training validations."""
    return {
        "model": model.state_dict(),
        "step": step,
        "mean_err": metrics["mean_err"],
        "median_err": metrics["median_err"],
        "p10": metrics["acc10"],
        "val_loss": metrics["loss"],
        "point_consensus": metrics["point_consensus"],
        "max_points": int(args.max_points),
        "point_batch_mode": "variable_no_replacement",
        "aug_deg": float(args.aug_deg),
        "normal_convention": "oriented_toward_camera",
        "aggregation": "normalize(mean(normalize(point_vectors)))",
        "pooling": "radial_weighted_mean(exp(-beta*r^2) * normalized_point_vectors)",
        "point_loss_weight": float(args.point_loss_weight),
        "ball_radius_m": float(args.ball_radius_m),
        "radial_feature": "r = clamp(norm((point - center) / ball_radius_m), 0, 1)",
        "radial_weight_beta": float(args.radial_weight_beta),
        "seed": int(args.seed),
        "init_checkpoint": os.path.abspath(args.init_checkpoint) if args.init_checkpoint else None,
        "finetune": bool(args.finetune),
    }


def write_run_metadata(out_dir, args, train_count, val_count):
    """Record settings that are needed to reproduce a checkpoint."""
    payload = {
        "normal_convention": "oriented_toward_camera",
        "aggregation": "normalize(mean(normalize(point_vectors)))",
        "pooling": "radial_weighted_mean(exp(-beta*r^2) * normalized_point_vectors)",
        "ball_radius_m": args.ball_radius_m,
        "radial_feature": "r = clamp(norm((point - center) / ball_radius_m), 0, 1)",
        "radial_weight_beta": args.radial_weight_beta,
        "loss": {
            "patch": "1 - dot(patch_normal, target)",
            "point": "1 - dot(point_normal, target)",
            "point_weight": args.point_loss_weight,
        },
        "train_samples": train_count,
        "val_samples": val_count,
        "args": vars(args),
    }
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def resolve_max_points(max_points, npoints, default=1024):
    """Keep --npoints usable for old commands while making the value an upper bound."""
    if max_points is not None and npoints is not None:
        raise ValueError("pass only one of --max-points and the deprecated --npoints alias")
    value = max_points if max_points is not None else npoints
    value = default if value is None else value
    if value < 0:
        raise ValueError("--max-points must be >= 0; 0 keeps every prepared point")
    return int(value)


def split_indices(files, split_path):
    """Return train/validation indices from a generated, file-name based split JSON."""
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    train_names = set(split.get("train", []))
    val_names = set(split.get("val", []))
    overlap = train_names & val_names
    if overlap:
        raise ValueError(f"train/val overlap in {split_path}: {next(iter(overlap))}")
    file_names = [str(x) for x in files]
    train_idx = np.array([i for i, name in enumerate(file_names) if name in train_names], dtype=np.int64)
    val_idx = np.array([i for i, name in enumerate(file_names) if name in val_names], dtype=np.int64)
    unknown = (train_names | val_names) - set(file_names)
    if unknown:
        raise ValueError(f"{split_path} references {len(unknown)} files absent from labels")
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"{split_path} must contain non-empty train and val splits")
    return train_idx, val_idx


@torch.no_grad()
def evaluate(model, loader, dev, point_loss_weight, radial_weight_beta):
    """Evaluate the same oriented, patch-level objective used for selection."""
    model.eval(); errs = []; losses = []; consensuses = []
    for coord, radial_distance, offset, y, _, counts in loader:
        coord, feat, offset = to_msecnet(coord.to(dev), radial_distance.to(dev), offset.to(dev))
        y = y.to(dev)
        pp = model(coord, feat, offset)
        loss, _, _, prediction, mean_vectors = normal_losses(
            pp, y, counts, feat, point_loss_weight, radial_weight_beta
        )
        errs.append(oriented_angular_error_deg(prediction, y).cpu().numpy())
        losses.append(loss.cpu().numpy())
        consensuses.append(mean_vectors.norm(dim=1).cpu().numpy())
    e = np.concatenate(errs)
    return {
        "loss": float(np.concatenate(losses).mean()),
        "mean_err": float(e.mean()),
        "median_err": float(np.median(e)),
        "acc10": float((e <= 10).mean() * 100),
        "point_consensus": float(np.concatenate(consensuses).mean()),
    }


class TrainLogger:
    """Write rectangular CSV metrics and a dashboard for a single training run."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self.history = defaultdict(list)
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=METRIC_COLUMNS).writeheader()

    def log_train(self, step, loss, lr):
        self._append({"step": step, "train_loss": loss, "lr": lr})

    def log_val(self, step, metrics):
        self._append({
            "step": step,
            "val_loss": metrics["loss"],
            "val_mean_ang_err": metrics["mean_err"],
            "val_median_ang_err": metrics["median_err"],
            "val_acc10_pct": metrics["acc10"],
            "val_point_consensus": metrics["point_consensus"],
        })

    def _append(self, values):
        row = {name: "" for name in METRIC_COLUMNS}
        row.update(values)
        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=METRIC_COLUMNS).writerow(row)
        for key, value in values.items():
            self.history[key].append(int(value) if key == "step" else float(value))
        step = int(values["step"])
        if "train_loss" in values:
            self.history["train_loss_step"].append(step)
        if "lr" in values:
            self.history["lr_step"].append(step)
        if "val_mean_ang_err" in values:
            self.history["val_mean_ang_err_step"].append(step)

    def plot_dashboard(self, step, save_path=None):
        """2×2 training dashboard: loss | val ang err | acc10 | lr."""
        if not HAS_MPL:
            return None
        if save_path is None:
            save_path = os.path.join(self.out_dir, "dashboard.png")
        h = self.history
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"MSECNet Training — step {step}", fontsize=13, fontweight="bold")

        ax = axes[0, 0]
        train_steps = h.get("train_loss_step", h.get("step", []))
        losses = h.get("train_loss", [])
        if len(train_steps) == len(losses) and losses:
            ax.plot(train_steps, losses, alpha=0.25, color="tab:blue", linewidth=0.6, label="raw (per 50 steps)")
            if len(losses) >= 5:
                w = min(9, len(losses) - (len(losses) % 2 == 0))
                if w >= 3:
                    smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
                    smooth_steps = train_steps[w // 2: w // 2 + len(smooth)]
                    ax.plot(smooth_steps, smooth, color="tab:blue", linewidth=1.8, label=f"smooth (w={w})")
        ax.set_ylabel("Training loss")
        ax.set_xlabel("Step")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        val_steps = h.get("val_mean_ang_err_step", [])
        vals = h.get("val_mean_ang_err", [])
        if len(val_steps) == len(vals) and len(vals) > 0:
            ax.plot(val_steps, vals, "o-", color="tab:orange", markersize=3, linewidth=1.2)
            if len(vals) > 0:
                ax.axhline(y=min(vals), color="tab:orange", linestyle=":", alpha=0.5,
                           label=f"best={min(vals):.2f}°")
            ax.legend(fontsize=8)
        ax.set_ylabel("Mean Angle Error (°)")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        accs = h.get("val_acc10_pct", [])
        if len(val_steps) == len(accs) and len(accs) > 0:
            ax.plot(val_steps, accs, "s-", color="tab:green", markersize=3, linewidth=1.2)
            if len(accs) > 0:
                ax.axhline(y=max(accs), color="tab:green", linestyle=":", alpha=0.5,
                           label=f"best={max(accs):.0f}%")
            ax.legend(fontsize=8)
        ax.set_ylabel("Angle ≤10° (%)")
        ax.set_xlabel("Step")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        lrs = h.get("lr", [])
        lr_steps = h.get("lr_step", [])
        if len(lr_steps) == len(lrs) and len(lrs) > 0:
            ax.plot(lr_steps, lrs, color="tab:red", linewidth=1.5)
        ax.set_ylabel("Learning Rate")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return save_path


def main():
    ap = argparse.ArgumentParser(description="Train MSECNet for center-anchored ball normals")
    ap.add_argument("labels"); ap.add_argument("pcd_dir")
    ap.add_argument("--steps", type=int, default=15000); ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--max-points", type=int, default=None,
                    help="maximum points per cloud; only larger clouds are sampled without replacement (0 keeps all)")
    ap.add_argument("--npoints", type=int, default=None,
                    help="deprecated alias for --max-points; retained for old commands")
    ap.add_argument("--inlier", type=float, default=0.8); ap.add_argument("--agree", type=float, default=15)
    ap.add_argument("--ball-radius-m", type=float, default=0.08,
                    help="prepared sphere radius in meters; must match anchors_manual3d.json")
    ap.add_argument("--radial-weight-beta", type=float, default=2.0,
                    help="near-center weight exp(-beta*r^2); 0 disables radial loss weighting")
    ap.add_argument("--soft", action="store_true")
    ap.add_argument("--aug-deg", type=float, default=45.0)
    ap.add_argument("--lr", type=float, default=None,
                    help="peak learning rate (default: 5e-4; 5e-5 with --finetune)")
    ap.add_argument("--init-checkpoint", default=None,
                    help="checkpoint whose model weights initialize this run")
    ap.add_argument("--finetune", action="store_true",
                    help="require --init-checkpoint; reset optimizer and use cosine decay")
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--point-loss-weight", type=float, default=0.25,
                    help="weight of per-point loss; the remaining weight optimizes the pooled patch normal")
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--val-workers", type=int, default=6)
    ap.add_argument("--early-stop-patience", type=int, default=100,
                    help="validation checks without mean-angle improvement before stopping; 0 disables")
    ap.add_argument("--snapshot-every", type=int, default=1000,
                    help="save a historical checkpoint every N steps; 0 disables snapshots")
    ap.add_argument("--centers", default=os.path.join(ROOT, "shared", "knob_centers.json"),
                    help="JSON anchors; entries with center_3d use the human 3D rectangle center")
    ap.add_argument("--split", default=None,
                    help="optional JSON with train/val file lists; use for car-model-disjoint validation")
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt_msecnet"))
    a = ap.parse_args(); a.max_points = resolve_max_points(a.max_points, a.npoints)
    if a.finetune and not a.init_checkpoint:
        ap.error("--finetune requires --init-checkpoint")
    if a.init_checkpoint and not os.path.isfile(a.init_checkpoint):
        raise FileNotFoundError(a.init_checkpoint)
    a.lr = a.lr if a.lr is not None else (5e-5 if a.finetune else 5e-4)
    if a.steps < 1 or a.bs < 1 or a.val_every < 1:
        raise ValueError("--steps, --bs, and --val-every must be positive")
    if a.lr <= 0:
        raise ValueError("--lr must be positive")
    if a.ball_radius_m <= 0 or a.radial_weight_beta < 0:
        raise ValueError("--ball-radius-m must be positive and --radial-weight-beta must be >= 0")
    if not 0 <= a.point_loss_weight <= 1:
        raise ValueError("--point-loss-weight must be in [0, 1]")
    if a.workers < 0 or a.val_workers < 0 or a.early_stop_patience < 0:
        raise ValueError("worker counts and --early-stop-patience must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this MSECNet pointops implementation")
    if not os.path.isfile(a.labels) or not os.path.isdir(a.pcd_dir):
        raise FileNotFoundError("labels and pcd_dir must exist")
    if not os.path.isfile(a.centers):
        raise FileNotFoundError(a.centers)
    seed_everything(a.seed)
    os.makedirs(a.out, exist_ok=True); dev = "cuda"
    logger = TrainLogger(a.out)

    L = np.load(a.labels); inl = L["inlier_frac"]; agr = L["agree_deg"]
    if a.split:
        tr_idx, va_idx = split_indices(L["files"], a.split)
        fa, na = L["files"], L["normal"]
        files, normals = fa[tr_idx], na[tr_idx]
        vfiles, vnormals = fa[va_idx], na[va_idx]
        if a.soft:
            w_all = (np.clip(inl, 0, 1) * np.clip(1 - agr / 30.0, 0, 1)).astype(np.float32)
            weights = w_all[tr_idx]
        else:
            weights = None
    elif a.soft:
        fa, na = L["files"], L["normal"]
        w_all = (np.clip(inl, 0, 1) * np.clip(1 - agr / 30.0, 0, 1)).astype(np.float32)
        clean = np.where((inl >= a.inlier) & (agr <= a.agree))[0]
        rng = np.random.default_rng(0); rng.shuffle(clean); nval = min(300, len(clean) // 3)
        va_set = set(clean[:nval].tolist()); tr_idx = np.array([i for i in range(len(fa)) if i not in va_set])
        files, normals, weights = fa[tr_idx], na[tr_idx], w_all[tr_idx]
        vfiles, vnormals = fa[clean[:nval]], na[clean[:nval]]
    else:
        gate = (inl >= a.inlier) & (agr <= a.agree); f2 = L["files"][gate]; n2 = L["normal"][gate]
        rng = np.random.default_rng(0); perm = rng.permutation(len(f2)); f2, n2 = f2[perm], n2[perm]
        nval = max(100, len(f2) // 10); vfiles, vnormals = f2[:nval], n2[:nval]
        files, normals, weights = f2[nval:], n2[nval:], None
    print(
        f"MSECNet: train {len(files)} / val {len(vfiles)} "
        f"(max_points={a.max_points}, point_loss_weight={a.point_loss_weight}, seed={a.seed})",
        flush=True,
    )
    if len(files) < a.bs:
        raise ValueError(f"training split has {len(files)} samples but batch size is {a.bs}")

    with open(a.centers, encoding="utf-8") as f:
        KC = json.load(f)
    loader_generator = torch.Generator().manual_seed(a.seed)
    tr = DataLoader(
        BallNormalDS(files, normals, a.pcd_dir, a.max_points, True, KC, a.ball_radius_m, weights, a.aug_deg, a.seed),
        batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
        persistent_workers=a.workers > 0, pin_memory=True, worker_init_fn=seed_worker,
        generator=loader_generator, collate_fn=collate_variable_points,
    )
    va = DataLoader(
        BallNormalDS(vfiles, vnormals, a.pcd_dir, a.max_points, False, KC, a.ball_radius_m, seed=a.seed),
        batch_size=a.bs, shuffle=False, num_workers=a.val_workers,
        persistent_workers=a.val_workers > 0, pin_memory=True, worker_init_fn=seed_worker,
        collate_fn=collate_variable_points,
    )
    write_run_metadata(a.out, a, len(files), len(vfiles))

    cfg = config.load_cfg_from_cfg_file(os.path.join(HERE, "config", "msecnet_ball.yaml"))
    cfg.num_classes = 3
    model = MSECNet(cfg).to(dev)
    if a.init_checkpoint:
        init_checkpoint = torch.load(a.init_checkpoint, map_location="cpu", weights_only=False)
        if "model" not in init_checkpoint:
            raise KeyError(f"{a.init_checkpoint} is missing model weights")
        model.load_state_dict(init_checkpoint["model"], strict=True)
        print(
            f"loaded initial model weights from {a.init_checkpoint} "
            f"(checkpoint step={init_checkpoint.get('step', 'unknown')})",
            flush=True,
        )
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    if a.finetune:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr / 100)
        print(f"fine-tuning with fresh AdamW and cosine decay (lr={a.lr:.2e})", flush=True)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    best = float("inf"); checks_without_improvement = 0; rl = 0.0; last_loss = 0.0

    if a.finetune:
        metrics = evaluate(model, va, dev, a.point_loss_weight, a.radial_weight_beta)
        initial_payload = checkpoint_payload(model, 0, metrics, a)
        logger.log_val(0, metrics)
        logger.plot_dashboard(0)
        atomic_torch_save(initial_payload, os.path.join(a.out, "last.pt"))
        atomic_torch_save(initial_payload, os.path.join(a.out, "best.pt"))
        best = metrics["mean_err"]
        print(
            f"  [VAL step 0] mean_ang_err={metrics['mean_err']:.2f}deg "
            f"median={metrics['median_err']:.2f}deg <=10deg:{metrics['acc10']:.0f}% "
            f"consensus={metrics['point_consensus']:.3f} (val_loss {metrics['loss']:.4f})",
            flush=True,
        )
        print(f"  -> saved starting best mean_ang_err {best:.2f}deg", flush=True)

    it = iter(tr)
    import time as _t; t0 = _t.time()
    for step in range(1, a.steps + 1):
        try:
            coord, radial_distance, offset, y, w, counts = next(it)
        except StopIteration:
            it = iter(tr); coord, radial_distance, offset, y, w, counts = next(it)
        coord, feat, offset = to_msecnet(
            coord.to(dev, non_blocking=True), radial_distance.to(dev, non_blocking=True), offset.to(dev, non_blocking=True)
        )
        y = y.to(dev, non_blocking=True); w = w.to(dev, non_blocking=True)
        pp = model(coord, feat, offset)
        sample_loss, _, _, _, _ = normal_losses(
            pp, y, counts, feat, a.point_loss_weight, a.radial_weight_beta
        )
        loss = (w * sample_loss).sum() / (w.sum() + 1e-6)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        rl = 0.9 * rl + 0.1 * loss.item()
        if step % 100 == 0:
            lr_now = sched.get_last_lr()[0]
            last_loss = rl
            print(f"step {step}/{a.steps} loss={last_loss:.4f} lr={lr_now:.2e} "
                  f"({100*a.bs/(_t.time()-t0):.0f}/s)", flush=True)
            logger.log_train(step, last_loss, lr_now)
            rl = 0.0; t0 = _t.time()
        is_snapshot_step = a.snapshot_every > 0 and (step % a.snapshot_every == 0 or step == a.steps)
        if step % a.val_every == 0 or is_snapshot_step or step == a.steps:
            metrics = evaluate(model, va, dev, a.point_loss_weight, a.radial_weight_beta)
            lr_now = sched.get_last_lr()[0]
            print(
                f"  [VAL step {step}] mean_ang_err={metrics['mean_err']:.2f}deg "
                f"median={metrics['median_err']:.2f}deg <=10deg:{metrics['acc10']:.0f}% "
                f"consensus={metrics['point_consensus']:.3f} (val_loss {metrics['loss']:.4f})",
                flush=True,
            )
            logger.log_val(step, metrics)
            logger.plot_dashboard(step)
            ckpt_payload = checkpoint_payload(model, step, metrics, a)
            atomic_torch_save(ckpt_payload, os.path.join(a.out, "last.pt"))
            if is_snapshot_step:
                snapshot_dir = os.path.join(a.out, "snapshots")
                os.makedirs(snapshot_dir, exist_ok=True)
                atomic_torch_save(ckpt_payload, os.path.join(snapshot_dir, f"step_{step:06d}.pt"))
            if metrics["mean_err"] < best:
                best = metrics["mean_err"]
                checks_without_improvement = 0
                atomic_torch_save(ckpt_payload, os.path.join(a.out, "best.pt"))
                print(f"  -> new best mean_ang_err {best:.2f}deg", flush=True)
            else:
                checks_without_improvement += 1
            if a.early_stop_patience and checks_without_improvement >= a.early_stop_patience:
                print(f"early stop after {checks_without_improvement} validations without improvement", flush=True)
                break
    print(f"done. best mean_ang_err={best:.2f}deg", flush=True)


if __name__ == "__main__":
    main()
