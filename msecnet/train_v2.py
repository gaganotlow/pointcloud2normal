#!/usr/bin/env python3
"""MSECNet for cap-normal, OFFSET-NATIVE so it supports VARIABLE / NO point subsampling (--max-points 0 = all).
Supervise every point with the (broadcast) cap weak-label normal; aggregate per-point preds -> cap normal.
Env: components/05_msecnet/.venv. Usage: msecnet_train.py <labels> <pcd_dir> [--max-points 0] [--aug-deg 45] ..."""
import argparse
import csv
import json
import math
import os
import re
import sys
import time

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
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MROOT = os.path.join(HERE, "MSECNet")
sys.path.insert(0, os.path.join(MROOT, "model")); sys.path.insert(0, os.path.join(MROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "shared"))
from util import config            # noqa: E402
from architectures import MSECNet  # noqa: E402
import cap_patch                   # noqa: E402


METRIC_FIELDS = [
    "time", "step", "split", "loss", "ema_loss", "lr", "mean_ang_err",
    "median_ang_err", "p10", "steps_per_sec", "best_mean_ang_err"
]


def log_print(out_dir, msg):
    print(msg, flush=True)
    with open(os.path.join(out_dir, "train.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_metrics_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in METRIC_FIELDS})


def record_metric(out_dir, history, row):
    row = dict(row)
    row.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    history.append(row)
    append_metrics_csv(os.path.join(out_dir, "metrics.csv"), row)
    append_jsonl(os.path.join(out_dir, "metrics.jsonl"), row)


def set_optimizer_lr(opt, lr):
    for group in opt.param_groups:
        group["lr"] = float(lr)


def get_optimizer_lr(opt):
    return float(opt.param_groups[0]["lr"])


def cosine_lr(step, total_steps, base_lr, min_lr, warmup_steps, warmup_start_factor):
    step = int(step)
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    if warmup_steps > 0 and step <= warmup_steps:
        alpha = step / float(warmup_steps)
        return base_lr * (warmup_start_factor + alpha * (1.0 - warmup_start_factor))
    span = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / float(span), 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def safe_stem(name, max_len=90):
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    stem = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", stem, flags=re.UNICODE)
    return stem[:max_len] or "sample"


def plot_training_curves(history, out_dir):
    if not HAS_MPL:
        return
    train = [r for r in history if r.get("split") == "train"]
    val = [r for r in history if r.get("split") == "val"]
    if not train and not val:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=False)
    if train:
        xs = [int(r["step"]) for r in train]
        axes[0].plot(xs, [float(r["ema_loss"]) for r in train], label="ema_loss", color="#1f77b4")
        axes[0].plot(xs, [float(r["loss"]) for r in train], label="loss", color="#9ecae1", alpha=0.45)
        axes[0].set_ylabel("train loss")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
    if val:
        xs = [int(r["step"]) for r in val]
        axes[1].plot(xs, [float(r["mean_ang_err"]) for r in val], marker="o", label="mean", color="#d62728")
        axes[1].plot(xs, [float(r["median_ang_err"]) for r in val], marker="o", label="median", color="#ff7f0e")
        axes[1].set_ylabel("angle error (deg)")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        axes[2].plot(xs, [float(r["p10"]) for r in val], marker="o", label="<=10deg %", color="#2ca02c")
        axes[2].set_ylabel("validation %")
        axes[2].set_xlabel("step")
        axes[2].legend()
        axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=160)
    plt.close(fig)


def rand_rot(rs, max_deg=180.0):
    ax = rs.normal(size=3); ax /= np.linalg.norm(ax) + 1e-9; a = np.radians(rs.uniform(0, max_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)).astype(np.float32)


def load_patch_xyz(npz_file, pcd_dir, kc_dict, radius, max_points=2500):
    with np.load(os.path.join(pcd_dir, npz_file)) as d:
        xyz = d["xyz"][d["label"] == 1].astype(np.float32)
        kc = kc_dict.get(npz_file)
        if kc is not None and len(xyz) >= 120:
            _, pm = cap_patch.extract(xyz, d["K_norm"], int(d["w"]), int(d["h"]), kc, radius_frac=radius)
            if pm.sum() >= 80:
                xyz = xyz[pm]
        if len(xyz) == 0:
            xyz = d["xyz"].astype(np.float32)
    xyz = xyz - xyz.mean(0)
    xyz = xyz / (np.linalg.norm(xyz, axis=1).max() + 1e-9)
    if max_points > 0 and len(xyz) > max_points:
        rs = np.random.default_rng(0)
        xyz = xyz[rs.choice(len(xyz), max_points, replace=False)]
    return xyz


def plot_validation_error_summary(details, step, out_dir):
    if not HAS_MPL or not details:
        return
    vis_dir = os.path.join(out_dir, "val_vis")
    os.makedirs(vis_dir, exist_ok=True)
    errs = np.array([d["error_deg"] for d in details], dtype=np.float32)
    order = np.argsort(errs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(errs, bins=30, color="#4c78a8", alpha=0.85)
    axes[0].axvline(float(errs.mean()), color="#e45756", label=f"mean {errs.mean():.2f}")
    axes[0].axvline(float(np.median(errs)), color="#f58518", label=f"median {np.median(errs):.2f}")
    axes[0].set_xlabel("angle error (deg)")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(np.arange(len(errs)), errs[order], color="#54a24b")
    axes[1].set_xlabel("validation samples sorted by error")
    axes[1].set_ylabel("angle error (deg)")
    axes[1].grid(alpha=0.2)

    fig.suptitle(f"Validation error at step {step}")
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, f"step_{step:06d}_error_summary.png"), dpi=160)
    plt.close(fig)


def plot_sample_cloud_normal(detail, xyz, out_path, title=None):
    if not HAS_MPL:
        return
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pred = np.asarray(detail["pred"], dtype=np.float32)
    target = np.asarray(detail["target"], dtype=np.float32)
    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=2, alpha=0.35, c=xyz[:, 2], cmap="viridis")
    ax.quiver(0, 0, 0, target[0], target[1], target[2], length=0.7, color="#2ca02c", linewidth=2, label="target")
    ax.quiver(0, 0, 0, pred[0], pred[1], pred[2], length=0.7, color="#d62728", linewidth=2, label="pred")
    ax.set_title(title or f"validation sample\nerr={detail['error_deg']:.2f}deg")
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_validation_visuals(details, step, out_dir, pcd_dir, kc_dict, radius, num_samples):
    if not details or num_samples <= 0:
        return
    step_dir = os.path.join(out_dir, "val_vis", f"step_{step:06d}")
    os.makedirs(step_dir, exist_ok=True)
    ranked = sorted(details, key=lambda d: d["error_deg"], reverse=True)
    for rank, detail in enumerate(ranked[:num_samples], 1):
        name = detail["file"]
        stem = f"{rank:02d}_err_{detail['error_deg']:.2f}_{safe_stem(name)}"
        try:
            xyz = load_patch_xyz(name, pcd_dir, kc_dict, radius)
            title = f"rank {rank} err={detail['error_deg']:.2f}deg"
            plot_sample_cloud_normal(detail, xyz, os.path.join(step_dir, stem + "_normal.png"), title=title)
        except Exception as exc:
            print(f"WARNING: failed to visualize {name}: {exc}", flush=True)


class DS(Dataset):
    def __init__(self, files, normals, pcd_dir, kc, radius, train, max_points,
                 weights=None, aug_deg=45.0, return_name=False):
        self.files, self.normals, self.pcd_dir, self.kc = files, normals, pcd_dir, kc
        self.radius, self.train, self.maxp, self.weights, self.aug_deg = radius, train, max_points, weights, aug_deg
        self.return_name = return_name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(os.path.join(self.pcd_dir, self.files[i]))
        xyz = d["xyz"][d["label"] == 1].astype(np.float32); n = self.normals[i].astype(np.float32).copy()
        kc = self.kc.get(self.files[i])
        if kc is not None and len(xyz) >= 120:
            _, pm = cap_patch.extract(xyz, d["K_norm"], int(d["w"]), int(d["h"]), kc, radius_frac=self.radius)
            if pm.sum() >= 80:
                xyz = xyz[pm]
        if len(xyz) == 0:
            xyz = d["xyz"].astype(np.float32)
        rs = np.random.RandomState((i * 7919 + int(time.time() * 1000)) & 0x7fffffff if self.train else i)
        if self.maxp > 0 and len(xyz) > self.maxp:                 # cap only if above max; 0 = keep ALL
            xyz = xyz[rs.choice(len(xyz), self.maxp, replace=False)]
        xyz = xyz - xyz.mean(0); xyz = xyz / (np.linalg.norm(xyz, axis=1).max() + 1e-9)
        if self.train:
            R = rand_rot(rs, self.aug_deg); xyz = xyz @ R.T; n = R @ n
            xyz = xyz + rs.normal(0, 0.01, xyz.shape).astype(np.float32)
        n = n / (np.linalg.norm(n) + 1e-9)
        w = float(self.weights[i]) if self.weights is not None else 1.0
        sample = (xyz.astype(np.float32), n.astype(np.float32), np.float32(w))
        if self.return_name:
            return sample + (str(self.files[i]),)
        return sample


def collate(batch):
    coords = [torch.from_numpy(b[0]) for b in batch]
    counts = torch.tensor([len(c) for c in coords], dtype=torch.int64)
    coord = torch.cat(coords, 0).float()
    offset = torch.cumsum(counts, 0).int()
    normal = torch.from_numpy(np.stack([b[1] for b in batch])).float()
    w = torch.tensor([b[2] for b in batch]).float()
    if len(batch[0]) > 3:
        names = [b[3] for b in batch]
        return coord, offset, normal, w, counts, names
    return coord, offset, normal, w, counts


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    errs = []
    details = []
    for batch in loader:
        if len(batch) == 6:
            coord, offset, normal, w, counts, names = batch
        else:
            coord, offset, normal, w, counts = batch
            names = [None] * len(counts)
        coord = coord.to(dev); offset = offset.to(dev); normal = normal.to(dev)
        pp = F.normalize(model(coord, torch.zeros(coord.shape[0], 0, device=dev), offset), dim=1)
        idx = 0
        for b, c in enumerate(counts.tolist()):
            agg = F.normalize(pp[idx:idx + c].mean(0), dim=0); idx += c
            target = F.normalize(normal[b], dim=0)
            dot = float(torch.dot(agg, target).clamp(-1, 1))
            cos = abs(dot)
            err = float(np.degrees(np.arccos(np.clip(cos, 0, 1))))
            errs.append(err)
            pred = agg.detach().cpu().numpy().astype(float)
            target_np = target.detach().cpu().numpy().astype(float)
            if dot < 0:
                pred = -pred
            details.append({
                "file": names[b],
                "error_deg": err,
                "pred": pred.tolist(),
                "target": target_np.tolist(),
                "points": int(c),
            })
    e = np.array(errs)
    return e.mean(), np.median(e), (e <= 10).mean() * 100, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("pcd_dir")
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--max-points", type=int, default=0)   # 0 = NO subsampling (all points); else cap
    ap.add_argument("--inlier", type=float, default=0.8); ap.add_argument("--agree", type=float, default=15)
    ap.add_argument("--radius", type=float, default=0.3); ap.add_argument("--soft", action="store_true")
    ap.add_argument("--aug-deg", type=float, default=45.0); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--sched", choices=("onecycle", "cosine", "constant"), default="onecycle")
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--warmup-start-factor", type=float, default=0.1)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--onecycle-pct-start", type=float, default=0.05)
    ap.add_argument("--onecycle-div-factor", type=float, default=25.0)
    ap.add_argument("--onecycle-final-div-factor", type=float, default=10000.0)
    ap.add_argument("--tail-steps", type=int, default=0,
                    help="extra low-LR OneCycle steps after the main schedule")
    ap.add_argument("--tail-lr", type=float, default=1e-5)
    ap.add_argument("--tail-wd", type=float, default=None)
    ap.add_argument("--tail-pct-start", type=float, default=0.15)
    ap.add_argument("--tail-div-factor", type=float, default=10.0)
    ap.add_argument("--tail-final-div-factor", type=float, default=10000.0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--vis-every", type=int, default=1000, help="0 disables per-sample validation images")
    ap.add_argument("--vis-samples", type=int, default=6)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop after this many validations without > --min-delta improvement")
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=0.0)
    ap.add_argument("--save-last", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt_msecnet_v2"))
    a = ap.parse_args()

    if a.sched == "cosine":
        if a.min_lr < 0 or a.min_lr > a.lr:
            raise ValueError("--min-lr must be in [0, --lr] for --sched cosine")
        if not 0 < a.warmup_start_factor <= 1:
            raise ValueError("--warmup-start-factor must be in (0, 1]")
    if not 0 < a.onecycle_pct_start <= 1:
        raise ValueError("--onecycle-pct-start must be in (0, 1]")
    if a.tail_steps < 0:
        raise ValueError("--tail-steps must be >= 0")
    if a.tail_steps > 0 and a.tail_lr <= 0:
        raise ValueError("--tail-lr must be > 0 when --tail-steps > 0")

    os.makedirs(a.out, exist_ok=True)
    for name in ("train.log", "metrics.csv", "metrics.jsonl"):
        path = os.path.join(a.out, name)
        if os.path.exists(path):
            os.remove(path)
    dev = a.device
    history = []
    run_start = time.strftime("%Y-%m-%d %H:%M:%S")
    log_print(a.out, f"Run started: {run_start}")
    log_print(a.out, "Command: " + " ".join(sys.argv))

    L = np.load(a.labels); inl = L["inlier_frac"]; agr = L["agree_deg"]
    if a.soft:
        fa, na = L["files"], L["normal"]; w_all = (np.clip(inl, 0, 1) * np.clip(1 - agr / 30.0, 0, 1)).astype(np.float32)
        clean = np.where((inl >= a.inlier) & (agr <= a.agree))[0]
        rng = np.random.default_rng(0); rng.shuffle(clean); nval = min(300, len(clean) // 3)
        va = set(clean[:nval].tolist()); tr = np.array([i for i in range(len(fa)) if i not in va])
        files, normals, weights = fa[tr], na[tr], w_all[tr]; vfiles, vnormals = fa[clean[:nval]], na[clean[:nval]]
    else:
        gate = (inl >= a.inlier) & (agr <= a.agree); f2 = L["files"][gate]; n2 = L["normal"][gate]
        rng = np.random.default_rng(0); p = rng.permutation(len(f2)); f2, n2 = f2[p], n2[p]
        nval = max(100, len(f2) // 10); vfiles, vnormals = f2[:nval], n2[:nval]; files, normals, weights = f2[nval:], n2[nval:], None
    log_print(a.out, f"MSECNet v2 (max_points={a.max_points}, 0=ALL): train {len(files)} / val {len(vfiles)} soft={a.soft}")

    KC = json.load(open(os.path.join(ROOT, "shared", "knob_centers.json")))
    write_json(os.path.join(a.out, "run_config.json"), {
        "started_at": run_start,
        "argv": sys.argv,
        "labels": a.labels,
        "pcd_dir": a.pcd_dir,
        "steps": a.steps,
        "tail_steps": a.tail_steps,
        "total_train_steps": int(a.steps + a.tail_steps),
        "batch_size": a.bs,
        "max_points": a.max_points,
        "inlier": a.inlier,
        "agree": a.agree,
        "radius": a.radius,
        "soft": bool(a.soft),
        "aug_deg": a.aug_deg,
        "lr": a.lr,
        "weight_decay": a.wd,
        "scheduler": a.sched,
        "warmup_steps": a.warmup_steps,
        "warmup_start_factor": a.warmup_start_factor,
        "min_lr": a.min_lr,
        "onecycle_pct_start": a.onecycle_pct_start,
        "onecycle_div_factor": a.onecycle_div_factor,
        "onecycle_final_div_factor": a.onecycle_final_div_factor,
        "tail_lr": a.tail_lr,
        "tail_weight_decay": a.tail_wd,
        "tail_pct_start": a.tail_pct_start,
        "tail_div_factor": a.tail_div_factor,
        "tail_final_div_factor": a.tail_final_div_factor,
        "patience": a.patience,
        "min_delta": a.min_delta,
        "grad_clip": a.grad_clip,
        "train_count": int(len(files)),
        "val_count": int(len(vfiles)),
        "val_files": [str(x) for x in vfiles],
    })
    tr = DataLoader(DS(files, normals, a.pcd_dir, KC, a.radius, True, a.max_points, weights, a.aug_deg),
                    batch_size=a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True, collate_fn=collate)
    vl = DataLoader(DS(vfiles, vnormals, a.pcd_dir, KC, a.radius, False, a.max_points, return_name=True),
                    batch_size=a.bs, shuffle=False, num_workers=4, collate_fn=collate)

    cfg = config.load_cfg_from_cfg_file(os.path.join(MROOT, "scripts/config/pcpnet/config.yaml")); cfg.num_classes = 3
    model = MSECNet(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = None
    if a.sched == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=a.lr, total_steps=a.steps, pct_start=a.onecycle_pct_start,
            div_factor=a.onecycle_div_factor, final_div_factor=a.onecycle_final_div_factor
        )
    elif a.sched == "constant":
        set_optimizer_lr(opt, a.lr)

    best = 999.0
    best_step = 0
    bad_vals = 0
    it = iter(tr)
    rl = 0.0
    t0 = time.time()
    total_train_steps = int(a.steps + a.tail_steps)
    tail_sched = None

    for step in range(1, total_train_steps + 1):
        if a.tail_steps > 0 and step == a.steps + 1:
            tail_wd = a.wd if a.tail_wd is None else a.tail_wd
            for group in opt.param_groups:
                group["weight_decay"] = float(tail_wd)
            tail_sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=a.tail_lr, total_steps=a.tail_steps,
                pct_start=a.tail_pct_start, div_factor=a.tail_div_factor,
                final_div_factor=a.tail_final_div_factor
            )
            log_print(a.out, f"Starting tail phase: steps={a.tail_steps} max_lr={a.tail_lr:.3e} wd={tail_wd}")

        in_tail = a.tail_steps > 0 and step > a.steps
        if a.sched == "cosine" and not in_tail:
            set_optimizer_lr(
                opt,
                cosine_lr(
                    step, a.steps, a.lr, a.min_lr,
                    min(a.warmup_steps, max(a.steps - 1, 0)),
                    a.warmup_start_factor
                )
            )
        try:
            coord, offset, normal, w, counts = next(it)
        except StopIteration:
            it = iter(tr); coord, offset, normal, w, counts = next(it)
        coord = coord.to(dev); offset = offset.to(dev); normal = normal.to(dev); w = w.to(dev)
        seg = torch.repeat_interleave(torch.arange(len(counts), device=dev), counts.to(dev))
        pp = F.normalize(model(coord, torch.zeros(coord.shape[0], 0, device=dev), offset), dim=1)
        tgt = normal[seg]; per = 1 - (pp * tgt).sum(1) ** 2; wpt = w[seg]
        loss = (wpt * per).sum() / (wpt.sum() + 1e-6)
        opt.zero_grad()
        loss.backward()
        if a.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        if in_tail:
            tail_sched.step()
        elif sched is not None:
            sched.step()
        rl = 0.9 * rl + 0.1 * loss.item()

        if a.log_every > 0 and (step % a.log_every == 0 or step == 1):
            steps_per_sec = step / max(time.time() - t0, 1e-9)
            lr_now = get_optimizer_lr(opt)
            record_metric(a.out, history, {
                "step": step,
                "split": "train",
                "loss": float(loss.item()),
                "ema_loss": float(rl),
                "lr": lr_now,
                "steps_per_sec": steps_per_sec,
                "best_mean_ang_err": best if best < 999.0 else "",
            })
            log_print(a.out, f"  [TRAIN step {step}] loss={loss.item():.5f} ema={rl:.5f} "
                      f"lr={lr_now:.3e} ({steps_per_sec:.2f}/s)")

        if step % a.val_every == 0 or step == total_train_steps:
            m, md, p10, details = evaluate(model, vl, dev)
            steps_per_sec = step / max(time.time() - t0, 1e-9)
            lr_now = get_optimizer_lr(opt)
            raw_best = m < best
            meaningful_best = best >= 999.0 or m < (best - a.min_delta)
            if raw_best:
                best = m
                best_step = step
            if meaningful_best:
                bad_vals = 0
            else:
                bad_vals += 1
            record_metric(a.out, history, {
                "step": step,
                "split": "val",
                "loss": float(loss.item()),
                "ema_loss": float(rl),
                "lr": lr_now,
                "mean_ang_err": float(m),
                "median_ang_err": float(md),
                "p10": float(p10),
                "steps_per_sec": steps_per_sec,
                "best_mean_ang_err": float(best),
            })
            log_print(a.out, f"  [VAL step {step}] mean_ang_err={m:.2f}deg median={md:.2f}deg "
                      f"<=10deg:{p10:.0f}% (loss {rl:.4f}, {steps_per_sec:.1f}/s)")
            val_path = os.path.join(a.out, "val_predictions", f"step_{step:06d}.json")
            write_json(val_path, {
                "step": int(step),
                "mean_ang_err": float(m),
                "median_ang_err": float(md),
                "p10": float(p10),
                "samples": details,
            })
            write_json(os.path.join(a.out, "val_predictions", "latest.json"), {
                "step": int(step),
                "mean_ang_err": float(m),
                "median_ang_err": float(md),
                "p10": float(p10),
                "samples": details,
            })
            plot_training_curves(history, a.out)
            plot_validation_error_summary(details, step, a.out)
            if a.vis_every > 0 and (step % a.vis_every == 0 or step == total_train_steps):
                save_validation_visuals(details, step, a.out, a.pcd_dir, KC, a.radius, a.vis_samples)
            ckpt_payload = {
                "model": model.state_dict(),
                "step": int(step),
                "mean_err": float(m),
                "median_err": float(md),
                "p10": float(p10),
                "max_points": int(a.max_points),
                "aug_deg": float(a.aug_deg),
                "scheduler": a.sched,
                "lr": float(a.lr),
                "weight_decay": float(a.wd),
                "tail_steps": int(a.tail_steps),
                "tail_lr": float(a.tail_lr),
            }
            if a.save_last:
                torch.save(ckpt_payload, os.path.join(a.out, "last.pt"))
            if raw_best:
                torch.save(ckpt_payload, os.path.join(a.out, "best.pt"))
                log_print(a.out, f"  -> new best mean_ang_err {best:.2f}deg")
            if a.patience > 0 and bad_vals >= a.patience:
                log_print(a.out, f"early stop at step {step}: no >{a.min_delta:.4f}deg improvement "
                          f"for {bad_vals} validations; best={best:.2f}deg at step {best_step}")
                break

    plot_training_curves(history, a.out)
    log_print(a.out, f"done. best mean_ang_err={best:.2f}deg at step {best_step}")


if __name__ == "__main__":
    main()
