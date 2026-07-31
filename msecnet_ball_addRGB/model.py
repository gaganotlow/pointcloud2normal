"""Late fusion model: MSECNet geometry encoder and a separate RGB image encoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


EPS = 1e-6


def radial_weighted_pool(features: torch.Tensor, counts: torch.Tensor, radial: torch.Tensor, beta: float) -> torch.Tensor:
    weights = torch.exp(-beta * radial.squeeze(1).square())
    return torch.stack([
        (item * item_weights[:, None]).sum(0) / item_weights.sum().clamp_min(EPS)
        for item, item_weights in zip(torch.split(features, counts.tolist()), torch.split(weights, counts.tolist()))
    ])


class RGBEncoder(nn.Module):
    def __init__(self, out_dim: int, pretrained: bool = False):
        super().__init__()
        # Avoid an implicit network download. Passing --image-pretrained requests torchvision's cached weights.
        if pretrained:
            from torchvision.models import ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(512, out_dim), nn.ReLU(inplace=True))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(image))


class PointRGBFusionNormalNet(nn.Module):
    """Keep point and image paths separate until global feature fusion."""

    def __init__(self, msecnet, geometry_dim: int = 128, image_dim: int = 128, pretrained_image: bool = False):
        super().__init__()
        self.geometry = msecnet
        # MSECNet's final classifier normally emits three point vectors. Its preceding tensor is the geometry feature.
        self.geometry.classifier = nn.Identity()
        self.point_head = nn.Linear(geometry_dim, 3)
        self.rgb = RGBEncoder(image_dim, pretrained=pretrained_image)
        self.image_head = nn.Linear(image_dim, 3)
        self.fusion_head = nn.Sequential(
            nn.Linear(geometry_dim + image_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, 3),
        )

    def forward(self, coord, radial, offset, image, counts, radial_weight_beta: float, rgb_dropout: float = 0.0):
        point_features = self.geometry(coord, radial, offset)
        geometry_feature = radial_weighted_pool(point_features, counts, radial, radial_weight_beta)
        point_vectors = self.point_head(point_features)
        rgb_feature = self.rgb(image)
        if self.training and rgb_dropout:
            keep = (torch.rand(len(rgb_feature), 1, device=rgb_feature.device) >= rgb_dropout).to(rgb_feature.dtype)
            rgb_feature = rgb_feature * keep
        fused_vector = self.fusion_head(torch.cat((geometry_feature, rgb_feature), dim=1))
        return {
            "point_vectors": point_vectors,
            "geometry_vector": radial_weighted_pool(point_vectors, counts, radial, radial_weight_beta),
            "image_vector": self.image_head(rgb_feature),
            "fused_vector": fused_vector,
        }


def normalized(vector: torch.Tensor) -> torch.Tensor:
    return F.normalize(vector, dim=1, eps=EPS)
