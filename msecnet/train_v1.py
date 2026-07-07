#!/usr/bin/env python3
"""MSECNet (edge-aware per-point normal net) adapted to OUR single-cap-normal task.
Input: cap patch points. MSECNet predicts PER-POINT normals; we supervise every point with the (single)
cap weak-label normal (broadcast), and at val AGGREGATE per-point preds -> one cap normal (axis error).
Reuses CapNormalDS for patch extraction + SO(3) aug. Env: components/05_msecnet/.venv. GPU.

Usage: python normal_net_msecnet.py <labels_npz> <pcd_dir> [--steps 15000] [--aug-deg 45] [--soft] ...
"""
import argparse
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
MROOT = os.path.join(HERE, "MSECNet")
sys.path.insert(0, os.path.join(MROOT, "model"))     # 'blocks'
sys.path.insert(0, os.path.join(MROOT, "scripts"))   # 'lib' (pointops), 'util'
sys.path.insert(0, os.path.join(ROOT, "shared"))     # cap_patch
from util import config                              # noqa: E402
from architectures import MSECNet                    # noqa: E402
from torch.utils.data import Dataset                 # noqa: E402
import cap_patch                                     # noqa: E402
import json                                          # noqa: E402
import time                                          # noqa: E402


def rand_rot(rs, max_deg=180.0):
    axis = rs.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
    ang = np.radians(rs.uniform(0, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return R.astype(np.float32)


class CapNormalDS(Dataset):              # data-only copy (no pointnet2 dep) — matches pose/normal_net.py
    def __init__(self, files, normals, pcd_dir, npoints, train, kc=None, radius=0.5, use_rgb=False, weights=None, aug_deg=180.0):
        self.files, self.normals, self.pcd_dir, self.npoints, self.train = files, normals, pcd_dir, npoints, train
        self.kc = kc or {}; self.radius = radius; self.weights = weights; self.aug_deg = aug_deg

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(os.path.join(self.pcd_dir, self.files[i]))
        xyz = d["xyz"][d["label"] == 1].astype(np.float32)
        n = self.normals[i].astype(np.float32).copy()
        kc = self.kc.get(self.files[i])
        if kc is not None and len(xyz) >= 120:
            _, pm = cap_patch.extract(xyz, d["K_norm"], int(d["w"]), int(d["h"]), kc, radius_frac=self.radius)
            if pm.sum() >= 80:
                xyz = xyz[pm]
        if len(xyz) == 0:
            xyz = d["xyz"].astype(np.float32)
        rs = np.random.RandomState((i * 7919 + int(time.time() * 1000)) & 0x7fffffff if self.train else i)
        idx = rs.choice(len(xyz), self.npoints, replace=(len(xyz) < self.npoints))
        xyz = xyz[idx]; xyz = xyz - xyz.mean(0); xyz = xyz / (np.linalg.norm(xyz, axis=1).max() + 1e-9)
        if self.train:
            R = rand_rot(rs, self.aug_deg); xyz = xyz @ R.T; n = R @ n
            xyz += rs.normal(0, 0.01, xyz.shape).astype(np.float32)
        n = n / (np.linalg.norm(n) + 1e-9)
        w = float(self.weights[i]) if self.weights is not None else 1.0
        return torch.from_numpy(xyz.T.copy()).float(), torch.zeros(3, len(xyz)), torch.from_numpy(n).float(), torch.tensor(w, dtype=torch.float32)


def to_msecnet(xyz_b):
    """(B,3,N) our batch -> (coord (B*N,3), feat (B*N,0), offset (B,) int32) for MSECNet."""
    B, _, N = xyz_b.shape
    coord = xyz_b.permute(0, 2, 1).reshape(B * N, 3).contiguous()
    feat = torch.zeros(B * N, 0, device=coord.device)
    offset = torch.arange(N, B * N + 1, N, dtype=torch.int32, device=coord.device)
    return coord, feat, offset, B, N


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); errs = []
    for x, ft, y, w in loader:
        coord, feat, offset, B, N = to_msecnet(x.to(dev))
        pp = model(coord, feat, offset).view(B, N, 3)        # per-point normals
        agg = F.normalize(pp.mean(1), dim=1)                 # aggregate -> cap normal
        cos = (agg * y.to(dev)).sum(1).abs().clamp(0, 1)
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
        if not HAS_MPL:
            return None
        if save_path is None:
            save_path = os.path.join(self.out_dir, "dashboard.png")
        h = self.history
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"MSECNet Training — step {step}", fontsize=13, fontweight="bold")

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

        ax = axes[0, 1]
        val_steps = [s for i, s in enumerate(h["step"]) if "val_mean_ang_err" in h and i < len(h["val_mean_ang_err"])]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("pcd_dir")
    ap.add_argument("--steps", type=int, default=15000); ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--inlier", type=float, default=0.8); ap.add_argument("--agree", type=float, default=15)
    ap.add_argument("--radius", type=float, default=0.3); ap.add_argument("--soft", action="store_true")
    ap.add_argument("--aug-deg", type=float, default=45.0)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt_msecnet"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"
    logger = TrainLogger(a.out)

    L = np.load(a.labels); inl = L["inlier_frac"]; agr = L["agree_deg"]
    if a.soft:
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
    print(f"MSECNet: train {len(files)} / val {len(vfiles)} (soft={a.soft})", flush=True)

    KC = json.load(open(os.path.join(ROOT, "shared", "knob_centers.json")))
    tr = DataLoader(CapNormalDS(files, normals, a.pcd_dir, a.npoints, True, KC, a.radius, False, weights, a.aug_deg),
                    batch_size=a.bs, shuffle=True, num_workers=10, drop_last=True, persistent_workers=True)
    va = DataLoader(CapNormalDS(vfiles, vnormals, a.pcd_dir, a.npoints, False, KC, a.radius, False),
                    batch_size=a.bs, shuffle=False, num_workers=6)

    cfg = config.load_cfg_from_cfg_file(os.path.join(MROOT, "scripts/config/pcpnet/config.yaml"))
    cfg.num_classes = 3
    model = MSECNet(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    best = 999.0; it = iter(tr); rl = 0.0; last_loss = 0.0
    import time as _t; t0 = _t.time()
    for step in range(1, a.steps + 1):
        try:
            x, ft, y, w = next(it)
        except StopIteration:
            it = iter(tr); x, ft, y, w = next(it)
        coord, feat, offset, B, N = to_msecnet(x.to(dev))
        y = y.to(dev); w = w.to(dev)
        pp = model(coord, feat, offset).view(B, N, 3)        # per-point pred
        tgt = y[:, None, :].expand(B, N, 3)                  # broadcast cap normal to all points
        per = 1 - (F.normalize(pp, dim=2) * tgt).sum(2) ** 2  # (B,N) sign-invariant per-point
        loss = (w[:, None] * per).sum() / (w.sum() * N + 1e-6)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        rl = 0.9 * rl + 0.1 * loss.item()
        if step % 100 == 0:
            lr_now = sched.get_last_lr()[0]
            last_loss = rl
            print(f"step {step}/{a.steps} loss={last_loss:.4f} lr={lr_now:.2e} "
                  f"({100*a.bs/(_t.time()-t0):.0f}/s)", flush=True)
            logger.log_train(step, last_loss, lr_now)
            rl = 0.0; t0 = _t.time()
        if step % a.val_every == 0 or step == a.steps:
            m, md, p10 = evaluate(model, va, dev)
            lr_now = sched.get_last_lr()[0]
            print(f"  [VAL step {step}] mean_ang_err={m:.2f}deg median={md:.2f}deg  <=10deg:{p10:.0f}%  "
                  f"(loss {last_loss:.4f}, lr={lr_now:.2e})", flush=True)
            logger.log_val(step, m, md, p10)
            logger.plot_dashboard(step)
            torch.save({"model": model.state_dict(), "step": step, "mean_err": float(m)},
                       os.path.join(a.out, "last.pt"))
            if m < best:
                best = m
                torch.save({"model": model.state_dict(), "step": step, "mean_err": float(m)},
                           os.path.join(a.out, "best.pt"))
                print(f"  -> new best mean_ang_err {best:.2f}deg", flush=True)
    print(f"done. best mean_ang_err={best:.2f}deg", flush=True)


if __name__ == "__main__":
    main()
