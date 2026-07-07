#!/usr/bin/env python3
"""MSECNet for cap-normal, OFFSET-NATIVE so it supports VARIABLE / NO point subsampling (--max-points 0 = all).
Supervise every point with the (broadcast) cap weak-label normal; aggregate per-point preds -> cap normal.
Env: components/05_msecnet/.venv. Usage: msecnet_train.py <labels> <pcd_dir> [--max-points 0] [--aug-deg 45] ..."""
import argparse
import json
import os
import sys
import time

import numpy as np
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


def rand_rot(rs, max_deg=180.0):
    ax = rs.normal(size=3); ax /= np.linalg.norm(ax) + 1e-9; a = np.radians(rs.uniform(0, max_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)).astype(np.float32)


class DS(Dataset):
    def __init__(self, files, normals, pcd_dir, kc, radius, train, max_points, weights=None, aug_deg=45.0):
        self.files, self.normals, self.pcd_dir, self.kc = files, normals, pcd_dir, kc
        self.radius, self.train, self.maxp, self.weights, self.aug_deg = radius, train, max_points, weights, aug_deg

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
        return xyz.astype(np.float32), n.astype(np.float32), np.float32(w)


def collate(batch):
    coords = [torch.from_numpy(b[0]) for b in batch]
    counts = torch.tensor([len(c) for c in coords], dtype=torch.int64)
    coord = torch.cat(coords, 0).float()
    offset = torch.cumsum(counts, 0).int()
    normal = torch.from_numpy(np.stack([b[1] for b in batch])).float()
    w = torch.tensor([b[2] for b in batch]).float()
    return coord, offset, normal, w, counts


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); errs = []
    for coord, offset, normal, w, counts in loader:
        coord = coord.to(dev); offset = offset.to(dev)
        pp = F.normalize(model(coord, torch.zeros(coord.shape[0], 0, device=dev), offset), dim=1)
        idx = 0
        for b, c in enumerate(counts.tolist()):
            agg = F.normalize(pp[idx:idx + c].mean(0), dim=0); idx += c
            cos = float(abs(torch.dot(agg, normal[b].to(dev))).clamp(0, 1))
            errs.append(np.degrees(np.arccos(cos)))
    e = np.array(errs); return e.mean(), np.median(e), (e <= 10).mean() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("pcd_dir")
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--max-points", type=int, default=0)   # 0 = NO subsampling (all points); else cap
    ap.add_argument("--inlier", type=float, default=0.8); ap.add_argument("--agree", type=float, default=15)
    ap.add_argument("--radius", type=float, default=0.3); ap.add_argument("--soft", action="store_true")
    ap.add_argument("--aug-deg", type=float, default=45.0); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-every", type=int, default=1000); ap.add_argument("--out", default=os.path.join(HERE, "ckpt_msecnet_v2"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"

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
    print(f"MSECNet v2 (max_points={a.max_points}, 0=ALL): train {len(files)} / val {len(vfiles)} soft={a.soft}", flush=True)

    KC = json.load(open(os.path.join(ROOT, "shared", "knob_centers.json")))
    tr = DataLoader(DS(files, normals, a.pcd_dir, KC, a.radius, True, a.max_points, weights, a.aug_deg),
                    batch_size=a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True, collate_fn=collate)
    vl = DataLoader(DS(vfiles, vnormals, a.pcd_dir, KC, a.radius, False, a.max_points),
                    batch_size=a.bs, shuffle=False, num_workers=4, collate_fn=collate)

    cfg = config.load_cfg_from_cfg_file(os.path.join(MROOT, "scripts/config/pcpnet/config.yaml")); cfg.num_classes = 3
    model = MSECNet(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    best = 999.0; it = iter(tr); rl = 0.0; t0 = time.time()
    for step in range(1, a.steps + 1):
        try:
            coord, offset, normal, w, counts = next(it)
        except StopIteration:
            it = iter(tr); coord, offset, normal, w, counts = next(it)
        coord = coord.to(dev); offset = offset.to(dev); normal = normal.to(dev); w = w.to(dev)
        seg = torch.repeat_interleave(torch.arange(len(counts), device=dev), counts.to(dev))
        pp = F.normalize(model(coord, torch.zeros(coord.shape[0], 0, device=dev), offset), dim=1)
        tgt = normal[seg]; per = 1 - (pp * tgt).sum(1) ** 2; wpt = w[seg]
        loss = (wpt * per).sum() / (wpt.sum() + 1e-6)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); rl = 0.9 * rl + 0.1 * loss.item()
        if step % a.val_every == 0 or step == a.steps:
            m, md, p10 = evaluate(model, vl, dev)
            print(f"  [VAL step {step}] mean_ang_err={m:.2f}deg median={md:.2f}deg <=10deg:{p10:.0f}% (loss {rl:.4f}, {step/(time.time()-t0):.1f}/s)", flush=True)
            if m < best:
                best = m; torch.save({"model": model.state_dict()}, os.path.join(a.out, "best.pt"))
    print(f"done. best mean_ang_err={best:.2f}deg", flush=True)


if __name__ == "__main__":
    main()
