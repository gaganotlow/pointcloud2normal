#!/usr/bin/env python3
"""Shared: extract the LOCAL cap-face patch around the knob center (option 2).

The whole inner_cover cloud is contaminated by the recess rim + (on some caps) MoGe's central
bulge. The flat cap FACE around the knob is cleaner. We anchor at the knob center (OBB 2D center
back-projected to the cap surface) and keep cap points within radius_frac * cap_radius of it.
"""
import numpy as np


def knob_anchor(cap_xyz, K, W, H, kc, k=30):
    """3D knob-center anchor = median of the cap points whose projection is nearest the OBB knob pixel."""
    Z = np.clip(cap_xyz[:, 2], 1e-3, None)
    u = K[0, 0] * W * cap_xyz[:, 0] / Z + K[0, 2] * W
    v = K[1, 1] * H * cap_xyz[:, 1] / Z + K[1, 2] * H
    kx, ky = kc["cxf"] * W, kc["cyf"] * H
    d2 = (u - kx) ** 2 + (v - ky) ** 2
    near = np.argsort(d2)[:min(k, len(cap_xyz))]
    return np.median(cap_xyz[near], 0)


def cap_patch_mask(cap_xyz, knob3d, radius_frac=0.5, min_pts=80):
    """Boolean mask over cap_xyz selecting the local face patch; grows radius if too few points."""
    capR = np.percentile(np.linalg.norm(cap_xyz - np.median(cap_xyz, 0), axis=1), 90) + 1e-9
    d = np.linalg.norm(cap_xyz - knob3d, axis=1)
    f = radius_frac
    mask = d < f * capR
    while mask.sum() < min_pts and f < 1.25:
        f += 0.15
        mask = d < f * capR
    return mask


def extract(cap_xyz, K, W, H, kc, radius_frac=0.5, min_pts=80):
    """Convenience: returns (knob3d, patch_mask) for an (N,3) inner_cover cloud + knob frac-center."""
    knob3d = knob_anchor(cap_xyz, K, W, H, kc)
    return knob3d, cap_patch_mask(cap_xyz, knob3d, radius_frac, min_pts)
