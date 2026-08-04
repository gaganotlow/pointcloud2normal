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


def load_obb_detections(path: Path) -> dict[str, dict]:
    """Load detector OBBs keyed by the prepared cloud file name."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "msecnet_ball_rgb_obb_detections_v1":
        raise ValueError(f"unsupported OBB detection cache: {path}")
    detections = payload.get("detections")
    if not isinstance(detections, dict):
        raise ValueError(f"OBB detection cache has no detections mapping: {path}")
    for name, detection in detections.items():
        corners = np.asarray(detection.get("corners"), dtype=np.float32)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError(f"invalid OBB corners for {name} in {path}")
    return detections


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


def obb_rgb_crop(image: np.ndarray, detection: dict, crop_scale: float, output_size: int) -> np.ndarray:
    """Use an OBB for localization while preserving the camera image orientation.

    Rotating the crop would discard the image-plane orientation needed to predict
    a normal in the camera coordinate frame. The detector therefore supplies the
    center and scale only; the crop itself remains axis-aligned.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must have shape HxWx3")
    if crop_scale <= 0:
        raise ValueError("OBB crop scale must be positive")
    left, top, side = obb_crop_bounds(detection, crop_scale)
    right, bottom = left + side, top + side
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, right - image.shape[1]), max(0, bottom - image.shape[0])
    if pad_l or pad_t or pad_r or pad_b:
        image = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
    left += pad_l; right += pad_l; top += pad_t; bottom += pad_t
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("detector OBB crop is empty")
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def obb_crop_bounds(detection: dict, crop_scale: float) -> tuple[int, int, int]:
    """Return the unpadded camera-image bounds used by ``obb_rgb_crop``."""
    corners = np.asarray(detection["corners"], dtype=np.float32)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("invalid OBB corners")
    if crop_scale <= 0:
        raise ValueError("OBB crop scale must be positive")
    center = np.asarray([detection.get("cx", corners[:, 0].mean()), detection.get("cy", corners[:, 1].mean())], dtype=np.float32)
    width = float(detection.get("w", corners[:, 0].max() - corners[:, 0].min()))
    height = float(detection.get("h", corners[:, 1].max() - corners[:, 1].min()))
    half = max(16, int(np.ceil(crop_scale * max(width, height) / 2.0)))
    return int(np.floor(center[0])) - half, int(np.floor(center[1])) - half, 2 * half


def project_points_to_crop_grid(
    points: np.ndarray, k_norm: np.ndarray, width: int, height: int, left: int, top: int, side: int,
) -> np.ndarray:
    """Project camera-space points into an axis-aligned crop's grid-sample coordinates."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or side <= 0:
        raise ValueError("points must have shape Nx3 and crop side must be positive")
    depth = np.maximum(points[:, 2], NORMAL_EPS)
    u = k_norm[0, 0] * width * points[:, 0] / depth + k_norm[0, 2] * width
    v = k_norm[1, 1] * height * points[:, 1] / depth + k_norm[1, 2] * height
    # align_corners=False interprets -1/+1 as the outside crop boundaries.
    grid = np.stack(((2 * (u - left) + 1) / side - 1, (2 * (v - top) + 1) / side - 1), axis=1)
    return np.clip(grid, -1, 1).astype(np.float32)


def _order_obb_corners(corners: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float32)
    sums = corners.sum(axis=1)
    differences = corners[:, 0] - corners[:, 1]
    ordered = np.stack((
        corners[np.argmin(sums)], corners[np.argmax(differences)],
        corners[np.argmax(sums)], corners[np.argmin(differences)],
    ))
    if len(np.unique(ordered, axis=0)) != 4:
        raise ValueError("OBB corners do not describe four distinct vertices")
    return ordered.astype(np.float32)


def rectified_obb_rgb_crop(image: np.ndarray, detection: dict, crop_scale: float, output_size: int) -> np.ndarray:
    """Historical crop protocol retained for evaluating pre-fix checkpoints."""
    corners = np.asarray(detection["corners"], dtype=np.float32)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("invalid OBB corners")
    center = corners.mean(axis=0, keepdims=True)
    source = _order_obb_corners(center + (corners - center) * crop_scale)
    destination = np.array([
        [0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image, transform, (output_size, output_size), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


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
        obb_detections=None, obb_crop_scale=1.4, obb_crop_mode="camera_oriented",
        sampling_indices=None, sampling_seed=None,
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
        self.sampling_indices = np.arange(len(self.files), dtype=np.int64) if sampling_indices is None else np.asarray(sampling_indices, dtype=np.int64)
        if self.sampling_indices.shape != (len(self.files),):
            raise ValueError("sampling_indices must provide one source index per file")
        self.sampling_seed = self.seed if sampling_seed is None else int(sampling_seed)
        self.obb_detections = obb_detections
        self.obb_crop_scale = float(obb_crop_scale)
        self.obb_crop_mode = obb_crop_mode
        missing = [name for name in self.files if name not in source_images or not source_images[name].is_file()]
        if missing:
            raise FileNotFoundError(f"RGB source image missing for {len(missing)} samples; first: {missing[0]}")
        if self.obb_detections is not None:
            missing = [name for name in self.files if name not in self.obb_detections]
            if missing:
                raise ValueError(f"OBB detection missing for {len(missing)} samples; first: {missing[0]}")
        if self.obb_crop_scale <= 0:
            raise ValueError("OBB crop scale must be positive")
        if self.obb_crop_mode not in ("camera_oriented", "rectified"):
            raise ValueError(f"unsupported OBB crop mode: {self.obb_crop_mode}")

    def __len__(self):
        return len(self.files)

    def _random_state(self, index: int) -> np.random.Generator:
        if self.train:
            return np.random.default_rng()
        return np.random.default_rng(self.sampling_seed + int(self.sampling_indices[index]) * 7919)

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
        camera_xyz = xyz.copy()
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
        if self.obb_detections is None:
            crop = centered_rgb_crop(
                image, center, k_norm, width, height, self.ball_radius_m,
                self.image_crop_scale, self.image_size,
            )
            point_uv = np.zeros((len(coord), 2), dtype=np.float32)
        else:
            crop_fn = rectified_obb_rgb_crop if self.obb_crop_mode == "rectified" else obb_rgb_crop
            crop = crop_fn(image, self.obb_detections[file_name], self.obb_crop_scale, self.image_size)
            left, top, side = obb_crop_bounds(self.obb_detections[file_name], self.obb_crop_scale)
            point_uv = project_points_to_crop_grid(camera_xyz, k_norm, width, height, left, top, side)
        weight = float(self.weights[index]) if self.weights is not None else 1.0
        return (
            coord.astype(np.float32), radial_distance, normalize_rgb(crop, self.train, random_state),
            (normal / (np.linalg.norm(normal) + NORMAL_EPS)).astype(np.float32), np.float32(weight), file_name, point_uv,
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
        torch.cat([torch.from_numpy(item[6]) for item in batch]).float(),
    )
