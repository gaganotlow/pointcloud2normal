"""Read the existing ball dataset and derive an RGB crop without changing it."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


NORMAL_EPS = 1e-6
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def source_images_from_index(source_root: Path) -> dict[str, Path]:
    """Map an existing prepared cloud filename to its original RGB image."""
    index_path = source_root / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    mapping = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        mapping[f"{row['id']}.npz"] = source_root / row["image"]
    return mapping


def project_center(center: np.ndarray, k_norm: np.ndarray, width: int, height: int) -> tuple[float, float]:
    z = max(float(center[2]), NORMAL_EPS)
    return (
        float(k_norm[0, 0]) * width * float(center[0]) / z + float(k_norm[0, 2]) * width,
        float(k_norm[1, 1]) * height * float(center[1]) / z + float(k_norm[1, 2]) * height,
    )


def centered_rgb_crop(
    image: np.ndarray, center: np.ndarray, k_norm: np.ndarray, width: int, height: int,
    ball_radius_m: float, crop_scale: float, output_size: int,
) -> np.ndarray:
    """Crop around the projected 3D ball center; this requires no point/pixel pairing."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must have shape HxWx3")
    u, v = project_center(center, k_norm, width, height)
    z = max(float(center[2]), NORMAL_EPS)
    radius_px = max(
        float(k_norm[0, 0]) * width * ball_radius_m / z,
        float(k_norm[1, 1]) * height * ball_radius_m / z,
    )
    half = max(16, int(np.ceil(crop_scale * radius_px)))
    left, top = int(np.floor(u)) - half, int(np.floor(v)) - half
    right, bottom = left + 2 * half, top + 2 * half
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, right - image.shape[1]), max(0, bottom - image.shape[0])
    if pad_l or pad_t or pad_r or pad_b:
        image = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
    left += pad_l; right += pad_l; top += pad_t; bottom += pad_t
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("projected RGB crop is empty")
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def normalize_rgb(image: np.ndarray, train: bool, random_state: np.random.Generator) -> np.ndarray:
    """Use photometric augmentation only; spatial image transforms would break camera geometry."""
    value = image.astype(np.float32) / 255.0
    if train:
        contrast = random_state.uniform(0.85, 1.15)
        brightness = random_state.uniform(-0.08, 0.08)
        saturation = random_state.uniform(0.85, 1.15)
        gray = value.mean(axis=2, keepdims=True)
        value = (value - gray) * saturation + gray
        value = value * contrast + brightness
        value = np.clip(value, 0.0, 1.0)
    value = (value - RGB_MEAN) / RGB_STD
    return np.moveaxis(value, 2, 0).astype(np.float32)


class BallRGBNormalDataset(Dataset):
    """Existing center-ball geometry plus a separate, projected source-image crop."""

    def __init__(
        self, files, normals, pcd_dir, anchors, source_images, ball_radius_m, max_points,
        image_size, image_crop_scale, train, weights=None, aug_deg=0.0, seed=0,
    ):
        self.files = [str(file) for file in files]
        self.normals = np.asarray(normals, dtype=np.float32)
        self.pcd_dir = Path(pcd_dir)
        self.anchors = anchors
        self.source_images = source_images
        self.ball_radius_m = float(ball_radius_m)
        self.max_points = int(max_points)
        self.image_size = int(image_size)
        self.image_crop_scale = float(image_crop_scale)
        self.train = bool(train)
        self.weights = weights
        self.aug_deg = float(aug_deg)
        self.seed = int(seed)
        missing = [name for name in self.files if name not in source_images or not source_images[name].is_file()]
        if missing:
            raise FileNotFoundError(f"RGB source image missing for {len(missing)} samples; first: {missing[0]}")

    def __len__(self):
        return len(self.files)

    def _random_state(self, index: int) -> np.random.Generator:
        if self.train:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + index * 7919)

    def __getitem__(self, index):
        file_name = self.files[index]
        anchor = self.anchors.get(file_name)
        if anchor is None or anchor.get("selection") != "manual_center_ball_patch":
            raise ValueError(f"{file_name} is not a prepared center-ball sample")
        if not np.isclose(float(anchor.get("ball_radius_m", 0)), self.ball_radius_m, rtol=0, atol=1e-8):
            raise ValueError(f"{file_name} has a different prepared ball radius")
        with np.load(self.pcd_dir / file_name) as cloud:
            valid = cloud["label"] == 1
            xyz = cloud["xyz"][valid].astype(np.float32)
            k_norm = cloud["K_norm"].astype(np.float32)
            width, height = int(cloud["w"]), int(cloud["h"])
        if not len(xyz):
            raise ValueError(f"{file_name} contains no ball points")
        random_state = self._random_state(index)
        if self.max_points and len(xyz) > self.max_points:
            xyz = xyz[random_state.choice(len(xyz), self.max_points, replace=False)]
        center = np.asarray(anchor["center_3d"], dtype=np.float32)
        coord = (xyz - center) / self.ball_radius_m
        normal = self.normals[index].copy()
        # Arbitrary 3D rotations are deliberately disabled by default because the RGB view is fixed.
        if self.train and self.aug_deg:
            axis = random_state.normal(size=3); axis /= np.linalg.norm(axis) + NORMAL_EPS
            angle = np.deg2rad(random_state.uniform(0, self.aug_deg))
            cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
            coord = coord @ rotation.T
            normal = rotation @ normal
            coord += random_state.normal(0, 0.01, coord.shape).astype(np.float32)
        radial_distance = np.minimum(np.linalg.norm(coord, axis=1, keepdims=True), 1.0).astype(np.float32)
        image = cv2.imread(str(self.source_images[file_name]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read {self.source_images[file_name]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop = centered_rgb_crop(
            image, center, k_norm, width, height, self.ball_radius_m,
            self.image_crop_scale, self.image_size,
        )
        weight = float(self.weights[index]) if self.weights is not None else 1.0
        return (
            coord.astype(np.float32), radial_distance, normalize_rgb(crop, self.train, random_state),
            (normal / (np.linalg.norm(normal) + NORMAL_EPS)).astype(np.float32), np.float32(weight), file_name,
        )


def collate_ball_rgb(batch):
    coords = [torch.from_numpy(item[0]) for item in batch]
    radial = [torch.from_numpy(item[1]) for item in batch]
    counts = torch.tensor([len(item) for item in coords], dtype=torch.int64)
    return (
        torch.cat(coords).float(), torch.cat(radial).float(), torch.cumsum(counts, dim=0).int(),
        torch.from_numpy(np.stack([item[2] for item in batch])).float(),
        torch.from_numpy(np.stack([item[3] for item in batch])).float(),
        torch.tensor([item[4] for item in batch]).float(), [item[5] for item in batch], counts,
    )
