#!/usr/bin/env python3
"""Point-cloud -> outward-normal regressor for the fuel-cap inner cover.

Input: the inner_cover sub-cloud (xyz), centered + unit-scaled (SCALE-INVARIANT, no intrinsics).
Output: a unit 3-vector = the whole-cap outward normal (camera frame). PointNet++ encoder + MLP head.
Trained on GEOMETRIC weak labels (RANSAC plane), gated to clean frames. The key augmentation is a
random SO(3) rotation applied to BOTH the cloud and the label, so the net learns shape->normal
(robust to noise/curvature/partial clouds) instead of memorizing the training orientation distribution.

Usage: python normalnet/train.py <labels_npz> <pcd_dir> [--steps 8000] [--bs 32] [--npoints 1024]
       [--inlier 0.5] [--agree 20] [--soft] [--out ckpt_normal]
"""
import argparse
import glob  # noqa: F401
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)                         # for pointnet2_utils.py
sys.path.insert(0, os.path.join(ROOT, "shared")) # for cap_patch.py
from pointnet2_utils import PointNetSetAbstraction  # noqa: E402
import cap_patch  # noqa: E402
import json  # noqa: E402


class NormalNet(nn.Module):
    def __init__(self, in_feat=0, dropout=0.3):  # in_feat = extra per-point channels (e.g. 3 for RGB)
        super().__init__()
        self.in_feat = in_feat
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 3 + in_feat, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512); self.bn1 = nn.BatchNorm1d(512); self.dp1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256); self.bn2 = nn.BatchNorm1d(256); self.dp2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, xyz, feat=None):           # xyz: (B,3,N); feat: (B,in_feat,N) or None
        B = xyz.shape[0]
        if self.in_feat == 0:
            feat = None
        l1x, l1p = self.sa1(xyz, feat)
        l2x, l2p = self.sa2(l1x, l1p)
        l3x, l3p = self.sa3(l2x, l2p)
        x = l3p.view(B, 1024)
        x = self.dp1(F.relu(self.bn1(self.fc1(x))))
        x = self.dp2(F.relu(self.bn2(self.fc2(x))))
        v = self.fc3(x)
        return F.normalize(v, dim=1)             # unit vector


def rand_rot(rs, max_deg=180.0):
    # random rotation with angle <= max_deg around a random axis (max_deg=180 ~ full SO(3))
    axis = rs.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
    ang = np.radians(rs.uniform(0, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)   # Rodrigues
    return R.astype(np.float32)


class CapNormalDS(Dataset):
    def __init__(self, files, normals, pcd_dir, npoints, train, kc=None, radius=0.5, use_rgb=False, weights=None, aug_deg=180.0, noise=True):
        self.files, self.normals, self.pcd_dir, self.npoints, self.train = files, normals, pcd_dir, npoints, train
        self.kc = kc or {}; self.radius = radius; self.use_rgb = use_rgb; self.weights = weights; self.aug_deg = aug_deg; self.noise = noise

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(os.path.join(self.pcd_dir, self.files[i]))
        inner = d["label"] == 1
        xyz = d["xyz"][inner].astype(np.float32)
        rgb = (d["rgb"][inner].astype(np.float32) / 255.0) if self.use_rgb else None
        n = self.normals[i].astype(np.float32).copy()
        kc = self.kc.get(self.files[i])               # option 2: local cap-face patch around knob
        if kc is not None and len(xyz) >= 120:
            _, pm = cap_patch.extract(xyz, d["K_norm"], int(d["w"]), int(d["h"]), kc, radius_frac=self.radius)
            if pm.sum() >= 80:
                xyz = xyz[pm]; rgb = rgb[pm] if rgb is not None else None
        if len(xyz) == 0:
            xyz = d["xyz"].astype(np.float32); rgb = (d["rgb"].astype(np.float32) / 255.0) if self.use_rgb else None
        rs = np.random.RandomState((i * 7919 + int(time.time() * 1000)) & 0x7fffffff if self.train else i)
        idx = rs.choice(len(xyz), self.npoints, replace=(len(xyz) < self.npoints))
        xyz = xyz[idx]; rgb = rgb[idx] if rgb is not None else None
        xyz = xyz - xyz.mean(0)
        xyz = xyz / (np.linalg.norm(xyz, axis=1).max() + 1e-9)     # scale-invariant
        if self.train:
            R = rand_rot(rs, self.aug_deg)         # rotate cloud AND label together (rgb is appearance -> NOT rotated)
            xyz = xyz @ R.T
            n = R @ n
            if self.noise:
                xyz += rs.normal(0, 0.01, xyz.shape).astype(np.float32)
            if rgb is not None:                    # color jitter so the net doesn't memorize lighting/paint
                rgb = np.clip(rgb * rs.uniform(0.7, 1.3) + rs.normal(0, 0.04, rgb.shape).astype(np.float32), 0, 1)
            if self.noise and rs.rand() < 0.3:     # random partial (drop a spatial half) -> robustness
                d0 = rs.normal(size=3); d0 /= np.linalg.norm(d0)
                keep = (xyz @ d0) > np.percentile(xyz @ d0, rs.uniform(0, 35))
                if keep.sum() > 64:
                    xyz = xyz[keep]; rgb = rgb[keep] if rgb is not None else None
                    ridx = rs.choice(len(xyz), self.npoints, replace=True)
                    xyz = xyz[ridx]; rgb = rgb[ridx] if rgb is not None else None
        n = n / (np.linalg.norm(n) + 1e-9)
        feat = torch.from_numpy(rgb.T.copy()).float() if rgb is not None else torch.zeros(3, len(xyz))
        w = float(self.weights[i]) if self.weights is not None else 1.0
        return torch.from_numpy(xyz.T.copy()).float(), feat, torch.from_numpy(n).float(), torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); errs = []
    for x, ft, y, w in loader:
        p = model(x.to(dev), ft.to(dev))
        cos = (p * y.to(dev)).sum(1).abs().clamp(0, 1)   # AXIS error (sign-invariant)
        errs.append(torch.rad2deg(torch.arccos(cos)).cpu().numpy())
    e = np.concatenate(errs)
    return e.mean(), np.median(e), (e <= 10).mean() * 100


class TrainLogger:
    """CSV metrics + dashboard plot per training run."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self.history = defaultdict(list)
        self._csv_header_written = False

    def log_train(self, step, loss, lr):
        self._append(step, {"loss": f"{loss:.6f}", "lr": f"{lr:.2e}"})

    def log_val(self, step, mean_err, median_err, acc10):
        self._append(step, {
            "val_mean_ang_err": f"{mean_err:.3f}",
            "val_median_ang_err": f"{median_err:.3f}",
            "val_acc10_pct": f"{acc10:.1f}",
        })

    def _append(self, step, kv):
        kv["step"] = step
        keys = ["step"] + [k for k in kv if k != "step"]
        with open(self.csv_path, "a") as f:
            if not self._csv_header_written:
                f.write(",".join(keys) + "\n")
                self._csv_header_written = True
            f.write(",".join(str(kv[k]) for k in keys) + "\n")
        for k, v in kv.items():
            self.history[k].append(float(v) if k != "step" else int(v))

    def plot_dashboard(self, step, save_path=None):
        """2×2 training dashboard: loss | val ang err | acc10 | lr."""
        if save_path is None:
            save_path = os.path.join(self.out_dir, "dashboard.png")
        h = self.history
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"NormalNet Training — step {step}", fontsize=13, fontweight="bold")

        # Top-left: training loss (raw + smoothed)
        ax = axes[0, 0]
        train_steps = [s for i, s in enumerate(h["step"]) if "loss" in h and i < len(h["loss"])]
        losses = h.get("loss", [])
        if len(train_steps) == len(losses) and len(losses) > 0:
            ax.plot(train_steps, losses, alpha=0.25, color="tab:blue", linewidth=0.6, label="raw (per 50 steps)")
            if len(losses) >= 5:
                w = min(9, len(losses) - (len(losses) % 2 == 0))
                if w >= 3:
                    smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
                    smooth_steps = train_steps[w // 2: w // 2 + len(smooth)]
                    ax.plot(smooth_steps, smooth, color="tab:blue", linewidth=1.8, label=f"smooth (w={w})")
        ax.set_ylabel("Loss  (1 − cos²)")
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Top-right: validation mean angle error
        ax = axes[0, 1]
        val_steps = [s for i, s in enumerate(h["step"]) if "val_mean_ang_err" in h and i < len(h["val_mean_ang_err"])]
        vals = h.get("val_mean_ang_err", [])
        if len(val_steps) == len(vals) and len(vals) > 0:
            ax.plot(val_steps, vals, "o-", color="tab:orange", markersize=5, linewidth=1.5)
            if len(vals) > 0:
                ax.axhline(y=min(vals), color="tab:orange", linestyle=":", alpha=0.5,
                           label=f"best={min(vals):.2f}°")
            ax.legend(fontsize=8)
        ax.set_ylabel("Mean Angle Error (°)")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

        # Bottom-left: validation accuracy (≤10°)
        ax = axes[1, 0]
        accs = h.get("val_acc10_pct", [])
        if len(val_steps) == len(accs) and len(accs) > 0:
            ax.plot(val_steps, accs, "s-", color="tab:green", markersize=5, linewidth=1.5)
            if len(accs) > 0:
                ax.axhline(y=max(accs), color="tab:green", linestyle=":", alpha=0.5,
                           label=f"best={max(accs):.0f}%")
            ax.legend(fontsize=8)
        ax.set_ylabel("Angle ≤10° (%)")
        ax.set_xlabel("Step")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # Bottom-right: learning rate
        ax = axes[1, 1]
        lrs = h.get("lr", [])
        lr_steps = [s for i, s in enumerate(h["step"]) if "lr" in h and i < len(h["lr"])]
        if len(lr_steps) == len(lrs) and len(lrs) > 0:
            ax.plot(lr_steps, lrs, color="tab:red", linewidth=1.5)
        ax.set_ylabel("Learning Rate")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return save_path


class Tee:
    """Duplicate stdout to a file."""

    def __init__(self, path):
        self.file = open(path, "a", buffering=1)
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("pcd_dir")
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--inlier", type=float, default=0.5); ap.add_argument("--agree", type=float, default=20)
    ap.add_argument("--radius", type=float, default=0.5)   # cap-patch radius (match the label-gen radius)
    ap.add_argument("--rgb", action="store_true")          # fuse per-point RGB (xyz+rgb) into the encoder
    ap.add_argument("--soft", action="store_true")         # E2: train ALL frames confidence-weighted; val=clean held-out
    ap.add_argument("--aug-deg", type=float, default=180.0)  # max rotation-aug angle (180~full SO(3); try 45 / 0)
    ap.add_argument("--no-noise", action="store_true")       # disable jitter + partial-drop (overfit/debug)
    ap.add_argument("--dropout", type=float, default=0.3)     # head dropout
    ap.add_argument("--wd", type=float, default=1e-4)         # weight decay
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt_normal"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"

    # ---- logging ----
    tee = Tee(os.path.join(a.out, "train.log"))
    sys.stdout = tee
    logger = TrainLogger(a.out)

    L = np.load(a.labels)
    inl = L["inlier_frac"]; agr = L["agree_deg"]
    if a.soft:                                    # E2: train on ALL frames weighted by confidence; val = clean held-out
        fa, na = L["files"], L["normal"]
        w_all = (np.clip(inl, 0, 1) * np.clip(1 - agr / 30.0, 0, 1)).astype(np.float32)
        clean = np.where((inl >= a.inlier) & (agr <= a.agree))[0]
        rng = np.random.default_rng(0); rng.shuffle(clean)
        nval = min(300, len(clean) // 3); va_set = set(clean[:nval].tolist())
        tr_idx = np.array([i for i in range(len(fa)) if i not in va_set])
        files, normals, weights = fa[tr_idx], na[tr_idx], w_all[tr_idx]
        vfiles, vnormals = fa[clean[:nval]], na[clean[:nval]]
        print(f"SOFT curriculum: train {len(files)} (all, weighted) / val {len(vfiles)} (clean held-out "
              f"inlier>={a.inlier} agree<={a.agree})", flush=True)
    else:
        gate = (inl >= a.inlier) & (agr <= a.agree)
        f2 = L["files"][gate]; n2 = L["normal"][gate]
        rng = np.random.default_rng(0); perm = rng.permutation(len(f2)); f2, n2 = f2[perm], n2[perm]
        nval = max(100, len(f2) // 10)
        vfiles, vnormals = f2[:nval], n2[:nval]; files, normals, weights = f2[nval:], n2[nval:], None
        print(f"clean labels: {gate.sum()}/{len(L['files'])} (gate inlier>={a.inlier} agree<={a.agree}); "
              f"train {len(files)} / val {len(vfiles)}", flush=True)

    kcp = os.path.join(ROOT, "shared", "knob_centers.json")
    KC = json.load(open(kcp)) if os.path.exists(kcp) else {}
    print(f"knob_centers loaded: {len(KC)} (local-patch training {'ON' if KC else 'OFF'})", flush=True)
    tr = DataLoader(CapNormalDS(files, normals, a.pcd_dir, a.npoints, True, KC, a.radius, a.rgb, weights, a.aug_deg, not a.no_noise),
                    batch_size=a.bs, shuffle=True, num_workers=10, drop_last=True, persistent_workers=True)
    va = DataLoader(CapNormalDS(vfiles, vnormals, a.pcd_dir, a.npoints, False, KC, a.radius, a.rgb),
                    batch_size=a.bs, shuffle=False, num_workers=6)

    model = NormalNet(in_feat=3 if a.rgb else 0, dropout=a.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    best = 999.0; it = iter(tr); rl = 0.0; t0 = time.time()
    for step in range(1, a.steps + 1):
        try:
            x, ft, y, w = next(it)
        except StopIteration:
            it = iter(tr); x, ft, y, w = next(it)
        x, ft, y, w = x.to(dev), ft.to(dev), y.to(dev), w.to(dev)
        p = model(x, ft)
        per = 1 - (p * y).sum(1) ** 2             # per-sample sign-invariant AXIS loss (disk +/-n ambiguity)
        loss = (w * per).sum() / (w.sum() + 1e-6)  # E2: confidence-weighted soft curriculum
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step(); rl += loss.item()
        if step % 50 == 0:
            lr_now = sched.get_last_lr()[0]
            print(f"step {step}/{a.steps} loss={rl/50:.4f} lr={lr_now:.2e} "
                  f"({50*a.bs/(time.time()-t0):.0f}/s)", flush=True)
            logger.log_train(step, rl / 50, lr_now)
            rl = 0; t0 = time.time()
        if step % a.val_every == 0 or step == a.steps:
            mean_e, med_e, acc10 = evaluate(model, va, dev)
            print(f"  [VAL step {step}] mean_ang_err={mean_e:.2f}deg median={med_e:.2f}deg  <=10deg:{acc10:.0f}%", flush=True)
            logger.log_val(step, mean_e, med_e, acc10)
            dashboard_path = logger.plot_dashboard(step)
            print(f"  [DASHBOARD] {dashboard_path}", flush=True)
            torch.save({"model": model.state_dict(), "step": step, "mean_err": float(mean_e)},
                       os.path.join(a.out, "last.pt"))
            if mean_e < best:
                best = mean_e
                torch.save({"model": model.state_dict(), "step": step, "mean_err": float(mean_e)},
                           os.path.join(a.out, "best.pt"))
                print(f"  -> new best mean_ang_err {best:.2f}deg", flush=True)
            model.train()
    print(f"done. best mean_ang_err={best:.2f}deg", flush=True)


if __name__ == "__main__":
    main()
