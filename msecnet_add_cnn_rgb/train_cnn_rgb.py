#!/usr/bin/env python3
"""MSECNet + CNN/ResNet RGB fusion.

This variant trains end-to-end from RGB images and point clouds. It does not
depend on offline MoGe/DINO feature files, so the same checkpoint can be used
for edge-style RGB + point-cloud inference.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MROOT = os.path.join(HERE, "..", "msecnet", "MSECNet")
sys.path.insert(0, os.path.join(MROOT, "model"))
sys.path.insert(0, os.path.join(MROOT, "scripts"))
POINTOPS_ROOT = os.path.join(MROOT, "scripts", "lib", "pointops")
for path in glob.glob(os.path.join(POINTOPS_ROOT, "build", "lib.*")):
    sys.path.insert(0, path)
sys.path.insert(0, os.path.join(ROOT, "shared"))

from architectures import MSECNet  # noqa: E402
from util import config  # noqa: E402
import cap_patch  # noqa: E402


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
RGB_FEAT_DIM = 512
RGB_BACKBONES = ("light", "resnet18", "resnet34", "resnet50", "resnet101")
RESNET_STAGES = ("layer2", "layer3", "layer4")
RESNET_STAGE_CHANNELS = {
    "resnet18": {"layer2": 128, "layer3": 256, "layer4": 512},
    "resnet34": {"layer2": 128, "layer3": 256, "layer4": 512},
    "resnet50": {"layer2": 512, "layer3": 1024, "layer4": 2048},
    "resnet101": {"layer2": 512, "layer3": 1024, "layer4": 2048},
}
RESNET_WEIGHT_ENUMS = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
}


def rand_rot(rs, max_deg=180.0):
    ax = rs.normal(size=3)
    ax /= np.linalg.norm(ax) + 1e-9
    a = np.radians(rs.uniform(0, max_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)).astype(np.float32)


def load_rgb_image(npz_path, npz_data, image_dir=None):
    if "rgb" in npz_data:
        rgb = npz_data["rgb"]
        if rgb.ndim == 3 and rgb.shape[-1] == 3:
            return rgb

    if image_dir:
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        candidates = [os.path.join(image_dir, stem)]
        if "__" in stem:
            car, image_stem = stem.split("__", 1)
            candidates.append(os.path.join(image_dir, car, image_stem))
        for ext in IMAGE_EXTS:
            for prefix in candidates:
                img_path = prefix + ext
                if not os.path.exists(img_path):
                    continue
                try:
                    import cv2
                except ImportError as exc:
                    raise RuntimeError("--image-dir requires cv2") from exc
                bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"failed to read image: {img_path}")
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raise ValueError(
        f"{os.path.basename(npz_path)} does not contain full-image rgb (H,W,3); "
        "pass --image-dir when npz rgb is per-point color"
    )


def image_to_tensor(rgb, image_size):
    rgb = np.asarray(rgb)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 is required to resize RGB images") from exc
    if image_size > 0:
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1)
    if t.max() > 1.5:
        t = t / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def project_points_to_image_norm(xyz, K_norm, w, h, image_size):
    xyz = np.asarray(xyz, dtype=np.float32)
    K_norm = np.asarray(K_norm, dtype=np.float32)
    z = xyz[:, 2]
    u = K_norm[0, 0] * float(w) * xyz[:, 0] / (z + 1e-9) + K_norm[0, 2] * float(w)
    v = K_norm[1, 1] * float(h) * xyz[:, 1] / (z + 1e-9) + K_norm[1, 2] * float(h)
    valid = (z > 1e-6) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)

    if image_size > 0:
        u = u * (float(image_size) / max(float(w), 1.0))
        v = v * (float(image_size) / max(float(h), 1.0))
        ww = float(image_size)
        hh = float(image_size)
    else:
        ww = float(w)
        hh = float(h)

    x = 2.0 * (u / max(ww - 1.0, 1.0)) - 1.0
    y = 2.0 * (v / max(hh - 1.0, 1.0)) - 1.0
    grid = np.stack([x, y], axis=1).astype(np.float32)
    grid[~valid] = 2.0
    return grid


def offset_to_counts(offset):
    start = torch.cat([offset.new_zeros(1), offset[:-1]])
    return (offset - start).long()


def log_print(out_dir, msg):
    print(msg, flush=True)
    with open(os.path.join(out_dir, "train.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def next_output_dir(prefix):
    run_id = 1
    while True:
        path = f"{prefix}_{run_id:03d}"
        try:
            os.makedirs(path)
            return path
        except FileExistsError:
            pass
        run_id += 1


METRIC_FIELDS = [
    "time", "step", "split", "loss", "ema_loss", "lr", "mean_ang_err",
    "median_ang_err", "p10", "steps_per_sec", "best_mean_ang_err"
]


def append_metric(out_dir, history, row):
    row = dict(row)
    row.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    history.append(row)
    csv_path = os.path.join(out_dir, "metrics.csv")
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in METRIC_FIELDS})
    with open(os.path.join(out_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable, skip plots: {exc}", flush=True)
        return None


def plot_curves(history, out_dir):
    plt = try_import_matplotlib()
    if plt is None:
        return
    train = [r for r in history if r.get("split") == "train"]
    val = [r for r in history if r.get("split") == "val"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    if train:
        xs = [int(r["step"]) for r in train]
        axes[0].plot(xs, [float(r["ema_loss"]) for r in train], label="ema_loss")
        axes[0].plot(xs, [float(r["loss"]) for r in train], alpha=0.35, label="loss")
        axes[0].set_ylabel("train loss")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
    if val:
        xs = [int(r["step"]) for r in val]
        axes[1].plot(xs, [float(r["mean_ang_err"]) for r in val], marker="o", label="mean")
        axes[1].plot(xs, [float(r["median_ang_err"]) for r in val], marker="o", label="median")
        axes[1].set_ylabel("angle error (deg)")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        axes[2].plot(xs, [float(r["p10"]) for r in val], marker="o", label="<=10deg %")
        axes[2].set_ylabel("validation %")
        axes[2].set_xlabel("step")
        axes[2].legend()
        axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=160)
    plt.close(fig)


class DSWithCNNRGB(Dataset):
    def __init__(self, files, normals, pcd_dir, image_dir, kc, radius,
                 train, max_points, image_size, weights=None, aug_deg=0.0,
                 return_name=False):
        self.files = files
        self.normals = normals
        self.pcd_dir = pcd_dir
        self.image_dir = image_dir
        self.kc = kc
        self.radius = radius
        self.train = train
        self.maxp = max_points
        self.image_size = image_size
        self.weights = weights
        self.aug_deg = aug_deg
        self.return_name = return_name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        npz_path = os.path.join(self.pcd_dir, str(self.files[i]))
        d = np.load(npz_path)
        xyz = d["xyz"][d["label"] == 1].astype(np.float32)
        n = self.normals[i].astype(np.float32).copy()

        kc = self.kc.get(str(self.files[i]))
        if kc is not None and len(xyz) >= 120:
            _, pm = cap_patch.extract(xyz, d["K_norm"], int(d["w"]), int(d["h"]), kc, radius_frac=self.radius)
            if pm.sum() >= 80:
                xyz = xyz[pm]
        if len(xyz) == 0:
            xyz = d["xyz"].astype(np.float32)

        rs = np.random.RandomState((i * 7919 + int(time.time() * 1000)) & 0x7fffffff if self.train else i)
        if self.maxp > 0 and len(xyz) > self.maxp:
            idx = rs.choice(len(xyz), self.maxp, replace=False)
            xyz = xyz[idx]

        rgb = load_rgb_image(npz_path, d, self.image_dir)
        image = image_to_tensor(rgb, self.image_size)
        grid = project_points_to_image_norm(xyz, d["K_norm"], int(d["w"]), int(d["h"]), self.image_size)

        xyz = xyz - xyz.mean(0)
        xyz = xyz / (np.linalg.norm(xyz, axis=1).max() + 1e-9)
        if self.train:
            R = rand_rot(rs, self.aug_deg)
            xyz = xyz @ R.T
            n = R @ n
            xyz = xyz + rs.normal(0, 0.01, xyz.shape).astype(np.float32)

        n = n / (np.linalg.norm(n) + 1e-9)
        w = float(self.weights[i]) if self.weights is not None else 1.0
        sample = (xyz.astype(np.float32), n.astype(np.float32), np.float32(w), image, grid)
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
    image = torch.stack([b[3] for b in batch], 0).float()
    grid = torch.from_numpy(np.concatenate([b[4] for b in batch], axis=0)).float()
    if len(batch[0]) > 5:
        names = [b[5] for b in batch]
        return coord, offset, normal, w, counts, image, grid, names
    return coord, offset, normal, w, counts, image, grid


class ConvBNAct(nn.Sequential):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )


class LightRGBEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            ConvBNAct(3, 32, stride=2),
            ConvBNAct(32, 64, stride=2),
            ConvBNAct(64, 96, stride=2),
            ConvBNAct(96, 128, stride=2),
            ConvBNAct(128, out_dim, stride=1),
        )

    def forward(self, image):
        return self.net(image)


def load_torchvision_resnet(name, pretrained):
    try:
        import torchvision.models as models
    except ImportError as exc:
        raise RuntimeError(
            f"--rgb-backbone {name} requires torchvision. Install torchvision in the training environment."
        ) from exc

    ctor = getattr(models, name)
    if not pretrained:
        try:
            return ctor(weights=None)
        except TypeError:
            return ctor(pretrained=False)

    weights_name = RESNET_WEIGHT_ENUMS[name]
    weights_enum = getattr(models, weights_name, None)
    if weights_enum is not None:
        return ctor(weights=weights_enum.DEFAULT)
    return ctor(pretrained=True)


class ResNetRGBEncoder(nn.Module):
    def __init__(self, name="resnet50", out_dim=128, stage="layer2", pretrained=True):
        super().__init__()
        if name not in RESNET_STAGE_CHANNELS:
            raise ValueError(f"unsupported ResNet backbone: {name}")
        if stage not in RESNET_STAGES:
            raise ValueError(f"unsupported ResNet stage: {stage}")

        resnet = load_torchvision_resnet(name, pretrained)
        layers = [
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        ]
        if stage in ("layer3", "layer4"):
            layers.append(resnet.layer3)
        if stage == "layer4":
            layers.append(resnet.layer4)

        in_dim = RESNET_STAGE_CHANNELS[name][stage]
        self.name = name
        self.stage = stage
        self.out_dim = out_dim
        self.features = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True),
        )

    def set_base_trainable(self, trainable):
        for p in self.features.parameters():
            p.requires_grad = bool(trainable)

    def freeze_batchnorm(self):
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def set_batchnorm_eval(self):
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, image):
        return self.proj(self.features(image))


def build_rgb_encoder(backbone, out_dim, pretrained=True, resnet_stage="layer2"):
    if backbone == "light":
        return LightRGBEncoder(out_dim)
    if backbone.startswith("resnet"):
        return ResNetRGBEncoder(backbone, out_dim, stage=resnet_stage, pretrained=pretrained)
    raise ValueError(f"unsupported rgb backbone: {backbone}")


def build_optimizer(model, lr, weight_decay=1e-4, rgb_backbone_lr_mult=1.0):
    if hasattr(model.rgb_encoder, "features"):
        backbone_params = list(model.rgb_encoder.features.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        return torch.optim.AdamW(
            [
                {"params": other_params, "lr": lr},
                {"params": backbone_params, "lr": lr * float(rgb_backbone_lr_mult)},
            ],
            weight_decay=weight_decay,
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


class MSECNetWithCNNRGB(nn.Module):
    def __init__(self, cfg, cnn_dim=128, rgb_feat_dim=RGB_FEAT_DIM,
                 rgb_backbone="light", rgb_pretrained=False, resnet_stage="layer2"):
        super().__init__()
        self.msecnet = MSECNet(cfg)
        geom_dim = self.msecnet.classifier[0].in_features
        self.geom_dim = geom_dim
        self.cnn_dim = int(cnn_dim)
        self.rgb_feat_dim = int(rgb_feat_dim)
        self.rgb_backbone = str(rgb_backbone)
        self.resnet_stage = str(resnet_stage)
        self.rgb_encoder = build_rgb_encoder(
            self.rgb_backbone,
            self.cnn_dim,
            pretrained=bool(rgb_pretrained),
            resnet_stage=self.resnet_stage,
        )
        self.rgb_proj = nn.Sequential(
            nn.Linear(2 * cnn_dim, rgb_feat_dim),
            nn.LayerNorm(rgb_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.gate = nn.Sequential(nn.Linear(geom_dim, 1), nn.Sigmoid())
        self.fusion_classifier = nn.Sequential(
            nn.Linear(geom_dim + rgb_feat_dim, geom_dim),
            nn.BatchNorm1d(geom_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(geom_dim, cfg.num_classes),
        )

    def set_rgb_backbone_trainable(self, trainable):
        if hasattr(self.rgb_encoder, "set_base_trainable"):
            self.rgb_encoder.set_base_trainable(trainable)
        else:
            for p in self.rgb_encoder.parameters():
                p.requires_grad = bool(trainable)

    def set_rgb_backbone_eval(self):
        if hasattr(self.rgb_encoder, "features"):
            self.rgb_encoder.features.eval()
        else:
            self.rgb_encoder.eval()

    def freeze_rgb_backbone_bn(self):
        if hasattr(self.rgb_encoder, "freeze_batchnorm"):
            self.rgb_encoder.freeze_batchnorm()

    def set_rgb_backbone_bn_eval(self):
        if hasattr(self.rgb_encoder, "set_batchnorm_eval"):
            self.rgb_encoder.set_batchnorm_eval()

    def sample_rgb_global(self, image, grid, counts):
        fmap = self.rgb_encoder(image)
        feats = []
        start = 0
        for b, c in enumerate(counts.tolist()):
            sample_grid = grid[start:start + c].to(fmap.device).view(1, c, 1, 2)
            sampled = F.grid_sample(
                fmap[b:b + 1],
                sample_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            pf = sampled[0, :, :, 0].T
            start += c
            if pf.numel() == 0:
                pooled = torch.zeros(2 * self.cnn_dim, dtype=fmap.dtype, device=fmap.device)
            else:
                pooled = torch.cat([pf.max(dim=0).values, pf.mean(dim=0)], dim=0)
            feats.append(pooled)
        return self.rgb_proj(torch.stack(feats, 0))

    def forward(self, p, x, o, image, grid):
        p_from_encoder = []
        x_from_encoder = []
        o_from_encoder = []
        side_output = []

        for block_i, block in enumerate(self.msecnet.encoder_blocks):
            if block_i in self.msecnet.encoder_skips:
                p_from_encoder.append(p)
                x_from_encoder.append(x)
                o_from_encoder.append(o)
                side_output.append([p, x, o])
            p, x, o = block(p, x, o)
        side_output.append([p, x, o])

        for block_i, block in enumerate(self.msecnet.decoder_blocks):
            if block_i in self.msecnet.decoder_upsample:
                p_dense = p_from_encoder.pop()
                x_dense = x_from_encoder.pop()
                o_dense = o_from_encoder.pop()
                p, x, o = block(p_dense, x_dense, o_dense, p, x, o)
            else:
                p, x, o = block(p, x, o)

        p_dense, x_dense, o_dense = side_output[0]
        ms_feat = x_dense
        for i in range(1, len(side_output[:self.msecnet.n_scale])):
            p_sp, x_sp, o_sp = side_output[i]
            from lib.pointops.functions import pointops
            interpolated = pointops.interpolation_flexible(
                p_sp, p_dense, x_sp, o_sp, o_dense,
                k=self.msecnet.nsample_interp,
                weight_type=self.msecnet.interp_weight_type,
            )
            ms_feat = torch.cat([ms_feat, interpolated], dim=1)

        ms_feat_new = self.msecnet.ms_fusion(p_dense, ms_feat, o_dense)[1]
        ms_edge = self.msecnet.edge_transfrom(p_dense, ms_feat_new, o_dense)[1]
        geom_feat = self.msecnet.ee(x, ms_edge)
        if geom_feat.shape[1] != self.geom_dim:
            raise RuntimeError(f"MSECNet feature dim changed: got {geom_feat.shape[1]}, expected {self.geom_dim}")

        counts = offset_to_counts(o)
        rgb_feat = self.sample_rgb_global(image, grid, counts)
        rgb_feat_expanded = rgb_feat.repeat_interleave(counts.to(rgb_feat.device), dim=0)
        alpha = self.gate(geom_feat)
        return self.fusion_classifier(torch.cat([geom_feat, alpha * rgb_feat_expanded], dim=1))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    errs = []
    details = []
    for batch in loader:
        if len(batch) == 8:
            coord, offset, normal, w, counts, image, grid, names = batch
        else:
            coord, offset, normal, w, counts, image, grid = batch
            names = [None] * len(counts)
        coord = coord.to(dev)
        offset = offset.to(dev)
        image = image.to(dev)
        grid = grid.to(dev)
        normal = normal.to(dev)
        pp = F.normalize(model(coord, torch.zeros(coord.shape[0], 0, device=dev), offset, image, grid), dim=1)
        idx = 0
        for b, c in enumerate(counts.tolist()):
            agg = F.normalize(pp[idx:idx + c].mean(0), dim=0)
            idx += c
            target = F.normalize(normal[b], dim=0)
            dot = float(torch.dot(agg, target).clamp(-1, 1))
            err = float(np.degrees(np.arccos(np.clip(abs(dot), 0, 1))))
            pred = agg.detach().cpu().numpy().astype(float)
            if dot < 0:
                pred = -pred
            errs.append(err)
            details.append({
                "file": names[b],
                "error_deg": err,
                "pred": pred.tolist(),
                "target": target.detach().cpu().numpy().astype(float).tolist(),
                "points": int(c),
            })
    e = np.array(errs)
    return e.mean(), np.median(e), (e <= 10).mean() * 100, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("pcd_dir")
    ap.add_argument("--image-dir", required=True, help="原始RGB图像目录")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--max-points", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--cnn-dim", type=int, default=128)
    ap.add_argument("--rgb-feat-dim", type=int, default=RGB_FEAT_DIM)
    ap.add_argument("--rgb-backbone", choices=RGB_BACKBONES, default="resnet50",
                    help="RGB feature extractor. Use 'light' to reproduce the old 5-layer CNN.")
    ap.add_argument("--resnet-stage", choices=RESNET_STAGES, default="layer2",
                    help="ResNet feature stage for grid sampling: layer2 is 1/8 resolution.")
    ap.add_argument("--no-rgb-pretrained", action="store_true",
                    help="Do not load ImageNet pretrained weights for ResNet backbones.")
    ap.add_argument("--freeze-rgb-backbone-steps", type=int, default=0,
                    help="Freeze the pretrained ResNet trunk for the first N steps.")
    ap.add_argument("--rgb-backbone-lr-mult", type=float, default=0.05,
                    help="Learning-rate multiplier for the ResNet trunk.")
    ap.add_argument("--no-freeze-rgb-bn", action="store_true",
                    help="Allow ResNet BatchNorm layers to update during training.")
    ap.add_argument("--inlier", type=float, default=0.8)
    ap.add_argument("--agree", type=float, default=15)
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--soft", action="store_true")
    ap.add_argument("--aug-deg", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    auto_out = args.out is None
    if auto_out:
        args.out = next_output_dir(os.path.join(HERE, f"ckpt_{args.rgb_backbone}"))
    else:
        os.makedirs(args.out, exist_ok=True)
    for name in ("train.log", "metrics.csv", "metrics.jsonl"):
        path = os.path.join(args.out, name)
        if os.path.exists(path):
            os.remove(path)
    history = []
    dev = args.device
    log_print(args.out, "Run started: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    log_print(args.out, "Command: " + " ".join(sys.argv))
    log_print(args.out, f"Output dir: {args.out} auto={auto_out}")

    L = np.load(args.labels)
    inl = L["inlier_frac"]
    agr = L["agree_deg"]
    if args.soft:
        fa, na = L["files"], L["normal"]
        w_all = (np.clip(inl, 0, 1) * np.clip(1 - agr / 30.0, 0, 1)).astype(np.float32)
        clean = np.where((inl >= args.inlier) & (agr <= args.agree))[0]
        rng = np.random.default_rng(0)
        rng.shuffle(clean)
        nval = min(300, len(clean) // 3)
        va = set(clean[:nval].tolist())
        tr_idx = np.array([i for i in range(len(fa)) if i not in va])
        files, normals, weights = fa[tr_idx], na[tr_idx], w_all[tr_idx]
        vfiles, vnormals = fa[clean[:nval]], na[clean[:nval]]
    else:
        gate = (inl >= args.inlier) & (agr <= args.agree)
        f2 = L["files"][gate]
        n2 = L["normal"][gate]
        rng = np.random.default_rng(0)
        p = rng.permutation(len(f2))
        f2, n2 = f2[p], n2[p]
        nval = max(100, len(f2) // 10)
        vfiles, vnormals = f2[:nval], n2[:nval]
        files, normals, weights = f2[nval:], n2[nval:], None

    rgb_pretrained = args.rgb_backbone != "light" and not args.no_rgb_pretrained
    freeze_rgb_bn = args.rgb_backbone != "light" and not args.no_freeze_rgb_bn
    log_print(args.out, f"MSECNet+CNN-RGB: train {len(files)} / val {len(vfiles)} soft={args.soft}")
    log_print(
        args.out,
        "RGB backbone: "
        f"{args.rgb_backbone} stage={args.resnet_stage} "
        f"pretrained={rgb_pretrained} cnn_dim={args.cnn_dim} rgb_feat_dim={args.rgb_feat_dim} "
        f"backbone_lr_mult={args.rgb_backbone_lr_mult} freeze_bn={freeze_rgb_bn}",
    )
    if args.aug_deg > 0:
        log_print(args.out, "WARNING: aug_deg > 0 rotates xyz/labels but not image features; use as ablation only.")
    KC = json.load(open(os.path.join(ROOT, "shared", "knob_centers.json")))
    write_json(os.path.join(args.out, "run_config.json"), vars(args) | {
        "train_count": int(len(files)),
        "val_count": int(len(vfiles)),
        "val_files": [str(x) for x in vfiles],
    })

    tr = DataLoader(
        DSWithCNNRGB(files, normals, args.pcd_dir, args.image_dir, KC, args.radius,
                     True, args.max_points, args.image_size, weights=weights,
                     aug_deg=args.aug_deg),
        batch_size=args.bs, shuffle=True, num_workers=8,
        drop_last=True, persistent_workers=True, collate_fn=collate,
    )
    vl = DataLoader(
        DSWithCNNRGB(vfiles, vnormals, args.pcd_dir, args.image_dir, KC, args.radius,
                     False, args.max_points, args.image_size, aug_deg=0.0,
                     return_name=True),
        batch_size=args.bs, shuffle=False, num_workers=4, collate_fn=collate,
    )

    cfg = config.load_cfg_from_cfg_file(os.path.join(MROOT, "scripts/config/pcpnet/config.yaml"))
    cfg.num_classes = 3
    model = MSECNetWithCNNRGB(
        cfg,
        args.cnn_dim,
        args.rgb_feat_dim,
        rgb_backbone=args.rgb_backbone,
        rgb_pretrained=rgb_pretrained,
        resnet_stage=args.resnet_stage,
    ).to(dev)
    if freeze_rgb_bn:
        model.freeze_rgb_backbone_bn()
        log_print(args.out, "Freeze ResNet BatchNorm running stats and affine parameters.")
    if args.freeze_rgb_backbone_steps > 0:
        model.set_rgb_backbone_trainable(False)
        log_print(args.out, f"Freeze RGB backbone trunk for first {args.freeze_rgb_backbone_steps} steps.")
    opt = build_optimizer(model, args.lr, weight_decay=1e-4, rgb_backbone_lr_mult=args.rgb_backbone_lr_mult)
    max_lr = [args.lr, args.lr * args.rgb_backbone_lr_mult] if hasattr(model.rgb_encoder, "features") else args.lr
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=args.steps, pct_start=0.05)

    best = 999.0
    it = iter(tr)
    rl = 0.0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.freeze_rgb_backbone_steps > 0 and step == args.freeze_rgb_backbone_steps + 1:
            model.set_rgb_backbone_trainable(True)
            if freeze_rgb_bn:
                model.freeze_rgb_backbone_bn()
            log_print(args.out, f"Unfreeze RGB backbone trunk at step {step}.")
        model.train()
        if freeze_rgb_bn:
            model.set_rgb_backbone_bn_eval()
        if args.freeze_rgb_backbone_steps > 0 and step <= args.freeze_rgb_backbone_steps:
            model.set_rgb_backbone_eval()

        try:
            coord, offset, normal, w, counts, image, grid = next(it)
        except StopIteration:
            it = iter(tr)
            coord, offset, normal, w, counts, image, grid = next(it)

        coord = coord.to(dev)
        offset = offset.to(dev)
        normal = normal.to(dev)
        w = w.to(dev)
        image = image.to(dev)
        grid = grid.to(dev)

        seg = torch.repeat_interleave(torch.arange(len(counts), device=dev), counts.to(dev))
        pp = F.normalize(model(coord, torch.zeros(coord.shape[0], 0, device=dev), offset, image, grid), dim=1)
        tgt = normal[seg]
        per = 1 - (pp * tgt).sum(1) ** 2
        wpt = w[seg]
        loss = (wpt * per).sum() / (wpt.sum() + 1e-6)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        rl = 0.9 * rl + 0.1 * loss.item()

        if args.log_every > 0 and (step % args.log_every == 0 or step == 1):
            sps = step / max(time.time() - t0, 1e-9)
            lr_now = float(sched.get_last_lr()[0])
            append_metric(args.out, history, {
                "step": step, "split": "train", "loss": float(loss.item()),
                "ema_loss": float(rl), "lr": lr_now, "steps_per_sec": sps,
                "best_mean_ang_err": best if best < 999 else "",
            })
            log_print(args.out, f"  [TRAIN step {step}] loss={loss.item():.5f} ema={rl:.5f} lr={lr_now:.3e} ({sps:.2f}/s)")

        if step % args.val_every == 0 or step == args.steps:
            m, md, p10, details = evaluate(model, vl, dev)
            is_best = m < best
            if is_best:
                best = m
            sps = step / max(time.time() - t0, 1e-9)
            lr_now = float(sched.get_last_lr()[0])
            append_metric(args.out, history, {
                "step": step, "split": "val", "loss": float(loss.item()),
                "ema_loss": float(rl), "lr": lr_now, "mean_ang_err": float(m),
                "median_ang_err": float(md), "p10": float(p10),
                "steps_per_sec": sps, "best_mean_ang_err": float(best),
            })
            log_print(args.out, f"  [VAL step {step}] mean_ang_err={m:.2f}deg median={md:.2f}deg <=10deg:{p10:.0f}%")
            write_json(os.path.join(args.out, "val_predictions", f"step_{step:06d}.json"), {
                "step": int(step), "mean_ang_err": float(m),
                "median_ang_err": float(md), "p10": float(p10), "samples": details,
            })
            write_json(os.path.join(args.out, "val_predictions", "latest.json"), {
                "step": int(step), "mean_ang_err": float(m),
                "median_ang_err": float(md), "p10": float(p10), "samples": details,
            })
            plot_curves(history, args.out)
            if is_best:
                torch.save({
                    "model": model.state_dict(),
                    "step": step,
                    "mean_err": float(m),
                    "median_err": float(md),
                    "p10": float(p10),
                    "cnn_dim": int(args.cnn_dim),
                    "rgb_feat_dim": int(args.rgb_feat_dim),
                    "rgb_backbone": str(args.rgb_backbone),
                    "rgb_pretrained": bool(rgb_pretrained),
                    "resnet_stage": str(args.resnet_stage),
                    "freeze_rgb_backbone_steps": int(args.freeze_rgb_backbone_steps),
                    "rgb_backbone_lr_mult": float(args.rgb_backbone_lr_mult),
                    "freeze_rgb_bn": bool(freeze_rgb_bn),
                    "image_size": int(args.image_size),
                    "aug_deg": float(args.aug_deg),
                    "arch": "msecnet_cnn_rgb",
                }, os.path.join(args.out, "best.pt"))

    plot_curves(history, args.out)
    log_print(args.out, f"done. best mean_ang_err={best:.2f}deg")


if __name__ == "__main__":
    main()
