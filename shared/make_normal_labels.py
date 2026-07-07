#!/usr/bin/env python3
"""Generate GEOMETRIC weak normal labels (+ confidence) for the cap clouds.

For each ROI cloud, take the inner_cover points, RANSAC-fit the cap plane → outward normal (oriented
toward the camera), and score confidence by (plane inlier fraction) and (agreement with the mean MoGe
per-point normal). High-confidence frames become the training labels for the point-cloud normal net;
low-confidence frames are the ones a human will later correct.

Usage: python make_normal_labels.py <pcd_dir> [--limit N] [--out labels.npz] [--radius 0.3]
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cap_patch  # noqa: E402

INNER = 1


def ransac_plane(pts, thresh, iters=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pts)
    best = None
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[i]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-12:
            continue
        nrm /= ln
        inl = np.abs((pts - p0) @ nrm) < thresh
        if best is None or inl.sum() > best[1].sum():
            best = (nrm, inl, p0)
    nrm, inl, _ = best
    c = pts[inl].mean(0)
    _, _, vt = np.linalg.svd(pts[inl] - c)
    nrm = vt[-1] / np.linalg.norm(vt[-1])
    inl = np.abs((pts - c) @ nrm) < thresh
    return nrm, c, inl


def main():
    pcd_dir = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else \
        os.path.join(HERE, "normal_labels.npz")
    files = sorted(glob.glob(os.path.join(pcd_dir, "*.npz")))
    if limit:
        files = files[:limit]
    patch = "--whole" not in sys.argv
    rad = float(sys.argv[sys.argv.index("--radius") + 1]) if "--radius" in sys.argv else 0.5
    KC = json.load(open(os.path.join(HERE, "knob_centers.json"))) if patch else {}
    print(f"{len(files)} clouds | mode={'LOCAL-PATCH around knob' if patch else 'WHOLE cap'} | knob_centers={len(KC)}", flush=True)
    rng = np.random.default_rng(0)

    keep_files, normals, confs, agrees, inlfracs, npts = [], [], [], [], [], []
    skipped_noknob = 0
    for k, f in enumerate(files):
        try:
            d = np.load(f)
        except Exception:
            continue
        lab = d["label"]; inner = lab == INNER
        cap_xyz = d["xyz"][inner].astype(np.float64)
        cap_nrm = d["normal"][inner].astype(np.float64)
        if patch:
            kc = KC.get(os.path.basename(f))
            if kc is None or len(cap_xyz) < 150:        # no knob detected -> no pose
                skipped_noknob += 1; continue
            knob3d, pm = cap_patch.extract(cap_xyz.astype(np.float32), d["K_norm"],
                                           int(d["w"]), int(d["h"]), kc, radius_frac=rad)
            if pm.sum() < 80:
                skipped_noknob += 1; continue
            xyz, nrm_moge = cap_xyz[pm], cap_nrm[pm]
        else:
            if inner.sum() < 200:
                continue
            xyz, nrm_moge = cap_xyz, cap_nrm
        n_used = len(xyz)
        if len(xyz) > 2500:
            idx = rng.choice(len(xyz), 2500, replace=False)
            xyz, nrm_moge = xyz[idx], nrm_moge[idx]
        span = np.linalg.norm(np.percentile(xyz, 97, 0) - np.percentile(xyz, 3, 0))
        thr = max(span * 0.02, 1e-5)
        n, c, inl = ransac_plane(xyz, thr)
        if n @ (-c) < 0:        # orient toward camera (origin)
            n = -n
        mn = nrm_moge.mean(0)
        if np.linalg.norm(mn) > 1e-6:
            mn = mn / np.linalg.norm(mn)
            if mn @ (-c) < 0:
                mn = -mn
            agree = float(np.degrees(np.arccos(np.clip(n @ mn, -1, 1))))
        else:
            agree = 180.0
        keep_files.append(os.path.basename(f))
        normals.append(n.astype(np.float32))
        confs.append(float(inl.mean()))
        agrees.append(agree)
        inlfracs.append(float(inl.mean()))
        npts.append(int(n_used))
        if (k + 1) % 1000 == 0:
            print(f"  {k+1}/{len(files)}  kept={len(keep_files)} skip_noknob={skipped_noknob}", flush=True)

    normals = np.array(normals); inlfracs = np.array(inlfracs); agrees = np.array(agrees)
    np.savez_compressed(out, files=np.array(keep_files), normal=normals,
                        inlier_frac=inlfracs, agree_deg=agrees, n_inner=np.array(npts))
    print(f"\nsaved {len(keep_files)} labels -> {out}")
    print(f"inlier_frac: med={np.median(inlfracs):.2f} p10={np.percentile(inlfracs,10):.2f} p90={np.percentile(inlfracs,90):.2f}")
    print(f"agree_deg(geom vs MoGe): med={np.median(agrees):.1f} p10={np.percentile(agrees,10):.1f} p90={np.percentile(agrees,90):.1f}")
    for thr_inl, thr_agree in [(0.5, 20), (0.6, 15), (0.7, 10)]:
        m = (inlfracs >= thr_inl) & (agrees <= thr_agree)
        print(f"  gate inlier>={thr_inl} & agree<={thr_agree}deg -> {m.sum()} clean frames ({100*m.mean():.0f}%)")


if __name__ == "__main__":
    main()
