#!/usr/bin/env python3
"""Train MSECNet to estimate one cap-plane normal from each local point cloud.

MSECNet predicts a normal for every point. Training broadcasts the sample's
manual normal target to its points; evaluation averages the point predictions
to one sign-invariant cap normal. Run in the ``point2normal`` Conda environment.
"""
import argparse
import csv
import os
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MROOT = os.path.join(HERE, "MSECNet")
sys.path.insert(0, os.path.join(MROOT, "model"))     # 'blocks'
sys.path.insert(0, os.path.join(MROOT, "scripts"))   # 'lib' (pointops), 'util'
sys.path.insert(0, HERE)                              # bundled cap_patch
from util import config                              # noqa: E402
from architectures import MSECNet                    # noqa: E402
from torch.utils.data import Dataset                 # noqa: E402
import cap_patch                                     # noqa: E402
import json                                          # noqa: E402
import time                                          # noqa: E402


METRIC_COLUMNS = (
    "step", "train_loss", "lr", "val_mean_ang_err",
    "val_median_ang_err", "val_acc10_pct",
)
ACTIVE_SOURCE_DATASET = "fuelcap_pass_20260803_10847"


def require_active_pseudo_obb_dataset(labels_path, pcd_dir, centers_path, split_path):
    """Reject accidental training on a superseded prepared dataset."""
    dataset_dir = Path(labels_path).resolve().parent
    expected_paths = (
        dataset_dir / "labels_manual3d.npz",
        dataset_dir / "clouds",
        dataset_dir / "anchors_manual3d.json",
        dataset_dir / "split_by_car_model.json",
    )
    if tuple(Path(path).resolve() for path in (labels_path, pcd_dir, centers_path, split_path)) != tuple(
        path.resolve() for path in expected_paths
    ):
        raise ValueError("labels, clouds, centers, and split must belong to one prepared pseudo-OBB dataset")
    metadata_path = dataset_dir / "dataset.json"
    if not metadata_path.is_file():
        raise ValueError(f"{dataset_dir} has no dataset.json provenance; prepare it from {ACTIVE_SOURCE_DATASET}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "pseudo_obb" or metadata.get("source_dataset") != ACTIVE_SOURCE_DATASET:
        raise ValueError(
            f"training only accepts pseudo-OBB data prepared from {ACTIVE_SOURCE_DATASET}; "
            f"got {metadata.get('source_dataset')!r}"
        )


def rand_rot(rs, max_deg=180.0):
    axis = rs.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
    ang = np.radians(rs.uniform(0, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return R.astype(np.float32)


class CapNormalDS(Dataset):
    def __init__(self, files, normals, pcd_dir, max_points, train, kc=None, radius=0.5, use_rgb=False, weights=None, aug_deg=180.0):
        self.files, self.normals, self.pcd_dir, self.max_points, self.train = files, normals, pcd_dir, max_points, train
        self.kc = kc or {}; self.radius = radius; self.weights = weights; self.aug_deg = aug_deg

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(os.path.join(self.pcd_dir, self.files[i]))
        xyz = d["xyz"][d["label"] == 1].astype(np.float32)
        n = self.normals[i].astype(np.float32).copy()
        kc = self.kc.get(self.files[i])
        if kc is not None:
            if kc.get("selection") in ("manual_innercap_sphere", "manual_pseudo_obb_patch"):
                # Prepared manual datasets are already deployment-shaped local patches.
                # Applying cap_patch_mask again would change their input definition.
                pm = None
            elif kc.get("selection") in ("manual_rect_prism", "manual_rect_sphere"):
                # Start from the human rectangle prism and draw the local sphere around its 3D center.
                pm = cap_patch.cap_patch_mask(
                    xyz, np.asarray(kc["center_3d"], dtype=np.float32), radius_frac=self.radius
                )
            elif "center_3d" in kc:
                # Human-labelled 3D rectangle center. Do not use the detector OBB.
                pm = cap_patch.cap_patch_mask(
                    xyz, np.asarray(kc["center_3d"], dtype=np.float32), radius_frac=self.radius
                )
            elif len(xyz) >= 120:
                _, pm = cap_patch.extract(xyz, d["K_norm"], int(d["w"]), int(d["h"]), kc, radius_frac=self.radius)
            else:
                pm = None
            if pm is not None and pm.sum() >= 80:
                xyz = xyz[pm]
        if len(xyz) == 0:
            xyz = d["xyz"].astype(np.float32)
        rs = np.random.RandomState((i * 7919 + int(time.time() * 1000)) & 0x7fffffff if self.train else i)
        if self.max_points > 0 and len(xyz) > self.max_points:
            xyz = xyz[rs.choice(len(xyz), self.max_points, replace=False)]
        xyz = xyz - xyz.mean(0); xyz = xyz / (np.linalg.norm(xyz, axis=1).max() + 1e-9)
        if self.train:
            R = rand_rot(rs, self.aug_deg); xyz = xyz @ R.T; n = R @ n
            xyz += rs.normal(0, 0.01, xyz.shape).astype(np.float32)
        n = n / (np.linalg.norm(n) + 1e-9)
        w = float(self.weights[i]) if self.weights is not None else 1.0
        return xyz.astype(np.float32), n.astype(np.float32), np.float32(w)


def collate_variable_points(batch):
    """Pack variable-length point clouds for MSECNet's offset-based interface."""
    coords = [torch.from_numpy(item[0]) for item in batch]
    counts = torch.tensor([len(coord) for coord in coords], dtype=torch.int64)
    return (
        torch.cat(coords, dim=0).float(),
        torch.cumsum(counts, dim=0).int(),
        torch.from_numpy(np.stack([item[1] for item in batch])).float(),
        torch.tensor([item[2] for item in batch]).float(),
        counts,
    )


def to_msecnet(coord, offset):
    """Add the empty feature tensor required by MSECNet to a packed point batch."""
    return coord, torch.zeros(coord.shape[0], 0, device=coord.device), offset


def aggregate_point_normals(point_normals, counts):
    """Return unnormalized and normalized mean normal vectors for each packed cloud."""
    means = torch.stack([points.mean(0) for points in torch.split(point_normals, counts.tolist())])
    return means, F.normalize(means, dim=1)


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
def evaluate(model, loader, dev):
    model.eval(); errs = []
    for coord, offset, y, _, counts in loader:
        coord, feat, offset = to_msecnet(coord.to(dev), offset.to(dev))
        _, agg = aggregate_point_normals(model(coord, feat, offset), counts)
        cos = (agg * y.to(dev)).sum(1).abs().clamp(0, 1)
        errs.append(torch.rad2deg(torch.arccos(cos)).cpu().numpy())
    e = np.concatenate(errs)
    return e.mean(), np.median(e), (e <= 10).mean() * 100


class TrainLogger:
    """Write rectangular CSV metrics and a dashboard for one training run."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self.history = defaultdict(list)
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=METRIC_COLUMNS).writeheader()

    def log_train(self, step, loss, lr):
        self._append({"step": step, "train_loss": loss, "lr": lr})

    def log_val(self, step, mean_err, median_err, acc10):
        self._append({
            "step": step,
            "val_mean_ang_err": mean_err,
            "val_median_ang_err": median_err,
            "val_acc10_pct": acc10,
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
            self.history["train_step"].append(step)
        if "lr" in values:
            self.history["lr_step"].append(step)
        if "val_mean_ang_err" in values:
            self.history["val_step"].append(step)

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
        train_steps = h.get("train_step", [])
        losses = h.get("train_loss", [])
        if len(train_steps) == len(losses) and len(losses) > 0:
            ax.plot(train_steps, losses, alpha=0.25, color="tab:blue", linewidth=0.6, label="raw (per 100 steps)")
            if len(losses) >= 5:
                w = min(9, len(losses) - (len(losses) % 2 == 0))
                if w >= 3:
                    smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
                    smooth_steps = train_steps[w // 2: w // 2 + len(smooth)]
                    ax.plot(smooth_steps, smooth, color="tab:blue", linewidth=1.8, label=f"smooth (w={w})")
        ax.set_ylabel("Training Loss (1 - cos^2)")
        ax.set_xlabel("Step")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        val_steps = h.get("val_step", [])
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


def write_run_metadata(out_dir, args, train_count, val_count):
    """Record the settings required to interpret a historical training run."""
    payload = {
        "normal_convention": "sign_invariant",
        "source_dataset": ACTIVE_SOURCE_DATASET,
        "aggregation": "normalize(mean(raw_point_vectors))",
        "loss": "1 - dot(normalize(point_prediction), target)^2",
        "train_samples": int(train_count),
        "val_samples": int(val_count),
        "args": vars(args),
    }
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("pcd_dir")
    ap.add_argument("--steps", type=int, default=15000); ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--max-points", type=int, default=None,
                    help="maximum points per cloud; only larger clouds are sampled without replacement (0 keeps all)")
    ap.add_argument("--npoints", type=int, default=None,
                    help="deprecated alias for --max-points; retained for old commands")
    ap.add_argument("--inlier", type=float, default=0.8); ap.add_argument("--agree", type=float, default=15)
    ap.add_argument("--radius", type=float, default=0.3); ap.add_argument("--soft", action="store_true")
    ap.add_argument("--aug-deg", type=float, default=45.0)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--snapshot-every", type=int, default=1000,
                    help="save a historical checkpoint every N steps; 0 disables snapshots")
    ap.add_argument("--centers", default=os.path.join(ROOT, "shared", "knob_centers.json"),
                    help="JSON anchors; entries with center_3d use the human 3D rectangle center")
    ap.add_argument("--split", default=None,
                    help="optional JSON with train/val file lists; use for car-model-disjoint validation")
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt_msecnet"))
    a = ap.parse_args(); a.max_points = resolve_max_points(a.max_points, a.npoints)
    if not a.split:
        raise ValueError("--split is required for the active 10847 training protocol")
    require_active_pseudo_obb_dataset(a.labels, a.pcd_dir, a.centers, a.split)
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
        f"(soft={a.soft}, max_points={a.max_points}, variable_batch=True)",
        flush=True,
    )
    write_run_metadata(a.out, a, len(files), len(vfiles))

    with open(a.centers, encoding="utf-8") as f:
        KC = json.load(f)
    tr = DataLoader(CapNormalDS(files, normals, a.pcd_dir, a.max_points, True, KC, a.radius, False, weights, a.aug_deg),
                    batch_size=a.bs, shuffle=True, num_workers=10, drop_last=True, persistent_workers=True,
                    collate_fn=collate_variable_points)
    va = DataLoader(CapNormalDS(vfiles, vnormals, a.pcd_dir, a.max_points, False, KC, a.radius, False),
                    batch_size=a.bs, shuffle=False, num_workers=6, collate_fn=collate_variable_points)

    cfg = config.load_cfg_from_cfg_file(os.path.join(MROOT, "scripts/config/pcpnet/config.yaml"))
    cfg.num_classes = 3
    model = MSECNet(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    best = 999.0; it = iter(tr); rl = 0.0; last_loss = 0.0
    import time as _t; t0 = _t.time()
    for step in range(1, a.steps + 1):
        try:
            coord, offset, y, w, counts = next(it)
        except StopIteration:
            it = iter(tr); coord, offset, y, w, counts = next(it)
        coord, feat, offset = to_msecnet(coord.to(dev), offset.to(dev))
        y = y.to(dev); w = w.to(dev)
        pp = model(coord, feat, offset)
        target_per_point = torch.repeat_interleave(y, counts.to(dev), dim=0)
        point_loss = 1 - (F.normalize(pp, dim=1) * target_per_point).sum(1) ** 2
        sample_loss = torch.stack([points.mean() for points in torch.split(point_loss, counts.tolist())])
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
            m, md, p10 = evaluate(model, va, dev)
            lr_now = sched.get_last_lr()[0]
            print(f"  [VAL step {step}] mean_ang_err={m:.2f}deg median={md:.2f}deg  <=10deg:{p10:.0f}%  "
                  f"(loss {last_loss:.4f}, lr={lr_now:.2e})", flush=True)
            logger.log_val(step, m, md, p10)
            logger.plot_dashboard(step)
            ckpt_payload = {
                "model": model.state_dict(),
                "step": step,
                "mean_err": float(m),
                "median_err": float(md),
                "p10": float(p10),
                "max_points": int(a.max_points),
                "point_batch_mode": "variable_no_replacement",
                "aug_deg": float(a.aug_deg),
                "source_dataset": ACTIVE_SOURCE_DATASET,
            }
            torch.save(ckpt_payload, os.path.join(a.out, "last.pt"))
            if is_snapshot_step:
                snapshot_dir = os.path.join(a.out, "snapshots")
                os.makedirs(snapshot_dir, exist_ok=True)
                torch.save(ckpt_payload, os.path.join(snapshot_dir, f"step_{step:06d}.pt"))
            if m < best:
                best = m
                torch.save(ckpt_payload, os.path.join(a.out, "best.pt"))
                print(f"  -> new best mean_ang_err {best:.2f}deg", flush=True)
    print(f"done. best mean_ang_err={best:.2f}deg", flush=True)


if __name__ == "__main__":
    main()
