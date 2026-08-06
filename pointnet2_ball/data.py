"""Shared dataset and metric helpers for the PointNet++ ball experiment."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parent.parent
ACTIVE_SOURCE_DATASET = "fuelcap_pass_20260803_10847"


def require_active_ball_dataset(labels_path, pcd_dir, centers_path, split_path):
    """Reject a mixed or superseded prepared center-ball dataset."""
    dataset_dir = Path(labels_path).resolve().parent
    expected = (
        dataset_dir / "labels_manual3d.npz",
        dataset_dir / "clouds",
        dataset_dir / "anchors_manual3d.json",
        dataset_dir / "split_by_generalization_group.json",
    )
    actual = tuple(Path(path).resolve() for path in (labels_path, pcd_dir, centers_path, split_path))
    if actual != tuple(path.resolve() for path in expected):
        raise ValueError("labels, clouds, centers, and split must belong to one prepared center-ball dataset")
    metadata_path = dataset_dir / "dataset.json"
    if not metadata_path.is_file():
        raise ValueError(f"{dataset_dir} has no dataset.json provenance")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "center_ball" or metadata.get("source_dataset") != ACTIVE_SOURCE_DATASET:
        raise ValueError(
            f"only center-ball data prepared from {ACTIVE_SOURCE_DATASET} is accepted; "
            f"got {metadata.get('source_dataset')!r}"
        )


def resolve_num_points(num_points, npoints, default=1024):
    """Resolve the fixed PointNet++ input cardinality, retaining --npoints."""
    if num_points is not None and npoints is not None:
        raise ValueError("pass only one of --num-points and --npoints")
    result = num_points if num_points is not None else npoints
    result = default if result is None else result
    if result < 16:
        raise ValueError("--num-points must be at least 16")
    return int(result)


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32)
    length = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or length < 1e-8:
        raise ValueError("normal must contain three finite, non-zero values")
    return vector / length


def rand_rot(rng, max_deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = np.deg2rad(rng.uniform(0, max_deg))
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=np.float32,
    )
    return (np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)).astype(np.float32)


def sample_fixed_points(xyz, num_points, rng):
    """Return exactly ``num_points`` points; repeat only when a ball is sparse."""
    if len(xyz) == 0:
        raise ValueError("point cloud has no usable points")
    indices = rng.choice(len(xyz), num_points, replace=len(xyz) < num_points)
    return xyz[indices]


class BallNormalDataset(Dataset):
    """Fixed-cardinality, center-anchored 8 cm balls for PointNet++."""

    def __init__(
        self, files, normals, pcd_dir, anchors, ball_radius_m, num_points,
        train, aug_deg=45.0, weights=None, seed=20260722, jitter_std=0.01,
    ):
        self.files = [str(value) for value in files]
        self.normals = np.asarray(normals, dtype=np.float32)
        self.pcd_dir = Path(pcd_dir)
        self.anchors = anchors
        self.ball_radius_m = float(ball_radius_m)
        self.num_points = int(num_points)
        self.train = bool(train)
        self.aug_deg = float(aug_deg)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)
        self.seed = int(seed)
        self.jitter_std = float(jitter_std)
        if len(self.files) != len(self.normals):
            raise ValueError("files and normals have different lengths")

    def __len__(self):
        return len(self.files)

    def _rng(self, index):
        # Worker RNG is seeded by seed_worker for training.  Validation must
        # remain identical across evaluations and processes.
        return np.random if self.train else np.random.default_rng(self.seed + index * 7919)

    def __getitem__(self, index):
        file_name = self.files[index]
        path = self.pcd_dir / file_name
        with np.load(path, allow_pickle=False) as cloud:
            xyz = cloud["xyz"].astype(np.float32)
            if "label" in cloud:
                xyz = xyz[cloud["label"] == 1]
        anchor = self.anchors.get(file_name)
        if not isinstance(anchor, dict) or anchor.get("selection") != "manual_center_ball_patch":
            raise ValueError(f"{file_name} is not a prepared center-ball sample")
        source_radius = float(anchor.get("ball_radius_m", 0))
        if not np.isclose(source_radius, self.ball_radius_m, rtol=0, atol=1e-8):
            raise ValueError(f"{file_name} was prepared at radius {source_radius}, not {self.ball_radius_m}")
        rng = self._rng(index)
        xyz = sample_fixed_points(xyz, self.num_points, rng)
        center = np.asarray(anchor["center_3d"], dtype=np.float32)
        if center.shape != (3,):
            raise ValueError(f"{file_name} has an invalid center_3d")
        xyz = (xyz - center) / self.ball_radius_m
        normal = normalized(self.normals[index])
        if self.train:
            rotation = rand_rot(rng, self.aug_deg)
            xyz = xyz @ rotation.T
            normal = rotation @ normal
            if self.jitter_std > 0:
                xyz += rng.normal(0, self.jitter_std, xyz.shape).astype(np.float32)
        radial_distance = np.minimum(np.linalg.norm(xyz, axis=1, keepdims=True), 1.0)
        weight = 1.0 if self.weights is None else float(self.weights[index])
        return xyz.astype(np.float32), radial_distance.astype(np.float32), normal.astype(np.float32), np.float32(weight)


def collate_fixed_points(batch):
    return (
        torch.from_numpy(np.stack([item[0] for item in batch])).float(),
        torch.from_numpy(np.stack([item[1] for item in batch])).float(),
        torch.from_numpy(np.stack([item[2] for item in batch])).float(),
        torch.tensor([item[3] for item in batch], dtype=torch.float32),
    )


def load_split_indices(files, split_path, split_name=None):
    """Read a named split, or return train/val index arrays when name is omitted."""
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    names = [str(name) for name in files]
    known = set(names)

    def indices(name):
        if name not in split:
            raise ValueError(f"{split_path} has no '{name}' split")
        selected = set(split[name])
        unknown = selected - known
        if unknown:
            raise ValueError(f"{split_path} references {len(unknown)} files absent from labels")
        result = np.array([index for index, file_name in enumerate(names) if file_name in selected], dtype=np.int64)
        if len(result) != len(selected):
            raise ValueError(f"duplicate file names in labels for split '{name}'")
        if not len(result):
            raise ValueError(f"split '{name}' is empty")
        return result

    if split_name is not None:
        return indices(split_name)
    train, val = indices("train"), indices("val")
    if set(train) & set(val):
        raise ValueError("train/val overlap")
    return train, val


def oriented_angular_error_deg(prediction, target):
    prediction = torch.nn.functional.normalize(prediction, dim=1, eps=1e-6)
    target = torch.nn.functional.normalize(target, dim=1, eps=1e-6)
    return torch.rad2deg(torch.arccos((prediction * target).sum(1).clamp(-1, 1)))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
