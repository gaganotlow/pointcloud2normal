"""Small, pure-PyTorch PointNet++ for one normal per local point-cloud ball."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def index_points(points, indices):
    """Gather ``points[B, N, C]`` at ``indices[B, ...]``."""
    batch = torch.arange(points.shape[0], device=points.device)
    batch = batch.view(points.shape[0], *([1] * (indices.ndim - 1))).expand_as(indices)
    return points[batch, indices]


def farthest_point_sample(xyz, npoint):
    """Deterministic FPS with the point nearest the ball center as first seed."""
    batch_size, total_points, _ = xyz.shape
    npoint = min(int(npoint), total_points)
    centroids = torch.empty(batch_size, npoint, dtype=torch.long, device=xyz.device)
    distances = torch.full((batch_size, total_points), float("inf"), device=xyz.device)
    farthest = xyz.square().sum(-1).argmin(dim=1)
    batch = torch.arange(batch_size, device=xyz.device)
    for index in range(npoint):
        centroids[:, index] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        distances = torch.minimum(distances, (xyz - centroid).square().sum(-1))
        farthest = distances.max(dim=1).indices
    return centroids


class SetAbstraction(nn.Module):
    """FPS + kNN grouping + shared MLP, the PointNet++ set-abstraction block."""

    def __init__(self, npoint, k, in_channels, mlp_channels):
        super().__init__()
        self.npoint = int(npoint)
        self.k = int(k)
        layers = []
        channels = in_channels + 3
        for output_channels in mlp_channels:
            layers.extend((nn.Conv2d(channels, output_channels, 1), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)))
            channels = output_channels
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features):
        # xyz [B, 3, N], features [B, C, N]
        points = xyz.transpose(1, 2).contiguous()
        sampled_indices = farthest_point_sample(points, self.npoint)
        centers = index_points(points, sampled_indices)
        k = min(self.k, points.shape[1])
        neighbors = torch.cdist(centers, points).topk(k, dim=-1, largest=False).indices
        grouped_xyz = index_points(points, neighbors) - centers.unsqueeze(2)
        grouped = grouped_xyz
        if features is not None:
            point_features = features.transpose(1, 2).contiguous()
            grouped = torch.cat((grouped_xyz, index_points(point_features, neighbors)), dim=-1)
        grouped = grouped.permute(0, 3, 1, 2).contiguous()
        new_features = self.mlp(grouped).max(dim=-1).values
        return centers.transpose(1, 2).contiguous(), new_features


class GlobalAbstraction(nn.Module):
    def __init__(self, in_channels, mlp_channels):
        super().__init__()
        layers = []
        channels = in_channels + 3
        for output_channels in mlp_channels:
            layers.extend((nn.Conv1d(channels, output_channels, 1), nn.BatchNorm1d(output_channels), nn.ReLU(inplace=True)))
            channels = output_channels
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features):
        return self.mlp(torch.cat((xyz, features), dim=1)).max(dim=-1).values


class PointNet2Normal(nn.Module):
    """Hierarchical PointNet++ encoder followed by one normalized normal head."""

    def __init__(self, sa1_points=256, sa2_points=64, sa1_k=32, sa2_k=32, dropout=0.3):
        super().__init__()
        self.model_args = {
            "sa1_points": int(sa1_points), "sa2_points": int(sa2_points),
            "sa1_k": int(sa1_k), "sa2_k": int(sa2_k), "dropout": float(dropout),
        }
        # The one input feature is the normalized point-to-center radius.
        self.sa1 = SetAbstraction(sa1_points, sa1_k, 1, (64, 64, 128))
        self.sa2 = SetAbstraction(sa2_points, sa2_k, 128, (128, 128, 256))
        self.global_sa = GlobalAbstraction(256, (256, 512, 1024))
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, xyz, radial_distance):
        # xyz [B, N, 3], radial_distance [B, N, 1]
        if xyz.ndim != 3 or xyz.shape[-1] != 3 or radial_distance.shape != (*xyz.shape[:2], 1):
            raise ValueError("expected xyz [B, N, 3] and radial_distance [B, N, 1]")
        xyz = xyz.transpose(1, 2).contiguous()
        features = radial_distance.transpose(1, 2).contiguous()
        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        return self.head(self.global_sa(xyz, features))

    def predict_normal(self, xyz, radial_distance):
        return F.normalize(self.forward(xyz, radial_distance), dim=1, eps=1e-6)


def build_model(model_args=None):
    return PointNet2Normal(**(model_args or {}))
