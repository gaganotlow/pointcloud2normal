"""Late fusion model: MSECNet geometry encoder and a separate RGB image encoder."""
from __future__ import annotations

import math
import os
from pathlib import Path

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


class ResNetRGBEncoder(nn.Module):
    def __init__(self, out_dim: int, pretrained: bool = False):
        super().__init__()
        if pretrained:
            from torchvision.models import ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(512, out_dim), nn.ReLU(inplace=True))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(image))


class DINOv2RGBEncoder(nn.Module):
    """Pretrained DINOv2 ViT-S/14 image features with a small trainable adapter."""

    FEATURE_DIM = 384

    def __init__(self, out_dim: int, unfreeze_blocks: int = 2):
        super().__init__()
        hub_dir = Path(torch.hub.get_dir())
        local_repo = Path(os.environ.get("DINOV2_REPO", hub_dir / "facebookresearch_dinov2_main"))
        if local_repo.is_dir():
            backbone = torch.hub.load(str(local_repo), "dinov2_vits14", source="local", pretrained=True)
        else:
            backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        unfreeze_blocks = max(0, min(int(unfreeze_blocks), len(self.backbone.blocks)))
        if unfreeze_blocks:
            for block in self.backbone.blocks[-unfreeze_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad_(True)
        self.trainable = unfreeze_blocks > 0
        self.projection = nn.Sequential(
            nn.LayerNorm(self.FEATURE_DIM), nn.Linear(self.FEATURE_DIM, out_dim), nn.GELU(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable:
            self.backbone.eval()
        return self

    def forward_with_patch_tokens(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.trainable:
            features = self.backbone.forward_features(image)
        else:
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone.forward_features(image)
        return self.projection(features["x_norm_clstoken"]), features["x_norm_patchtokens"]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features, _ = self.forward_with_patch_tokens(image)
        return features


class PointRGBFusionNormalNet(nn.Module):
    """Keep point and image paths separate until global feature fusion."""

    def __init__(self, msecnet, geometry_dim: int = 128, image_dim: int = 128,
                 pretrained_image: bool = True, image_backbone: str = "dino_vits14",
                 dino_unfreeze_blocks: int = 2, fusion_mode: str = "gated_residual",
                 geometry_mode: str = "feature_head", max_rgb_correction: float | None = None,
                 initial_gate: float = 0.018):
        super().__init__()
        self.geometry = msecnet
        self.geometry_mode = geometry_mode
        self.geometry_frozen = False
        if not 0 < initial_gate < 1:
            raise ValueError("initial_gate must be strictly between zero and one")
        self.initial_gate = float(initial_gate)
        # ``None`` retains the historical unconstrained residual used by V1/V2
        # checkpoints. New frozen-baseline runs set an explicit small limit.
        self.max_rgb_correction = None if max_rgb_correction is None else float(max_rgb_correction)
        if geometry_mode == "feature_head":
            # Historical RGB checkpoints use an independently initialized point head.
            self.geometry.classifier = nn.Identity()
            self.point_head = nn.Linear(geometry_dim, 3)
            self._geometry_features = None
        elif geometry_mode == "pretrained_point":
            # Preserve a trained MSECNet classifier exactly and capture its 128-D input for fusion.
            self.point_head = None
            self._geometry_features = None
            self.geometry.classifier.register_forward_pre_hook(self._capture_geometry_features)
        else:
            raise ValueError(f"unsupported geometry mode: {geometry_mode}")
        if image_backbone == "dino_vits14":
            self.rgb = DINOv2RGBEncoder(image_dim, unfreeze_blocks=dino_unfreeze_blocks)
        elif image_backbone == "resnet18":
            # Keep this direct module assignment so existing ResNet checkpoints
            # retain their historical rgb.features and rgb.projection key names.
            self.rgb = ResNetRGBEncoder(image_dim, pretrained=pretrained_image)
        else:
            raise ValueError(f"unsupported RGB backbone: {image_backbone}")
        self.image_head = nn.Linear(image_dim, 3)
        self.fusion_mode = fusion_mode
        if fusion_mode == "gated_residual":
            self.fusion_head = nn.Sequential(
                nn.Linear(geometry_dim + image_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, 3),
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(geometry_dim + image_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1),
            )
            # Geometry features can have a much larger scale than RGB features.
            # Zeroing the final weights makes the initial gate exactly controlled
            # by its bias instead of accidentally saturating from that scale.
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.constant_(self.fusion_gate[-1].bias, math.log(self.initial_gate / (1 - self.initial_gate)))
        elif fusion_mode == "point_aligned_residual":
            if image_backbone != "dino_vits14":
                raise ValueError("point_aligned_residual requires --image-backbone dino_vits14")
            self.point_visual_adapter = nn.Sequential(
                nn.LayerNorm(DINOv2RGBEncoder.FEATURE_DIM), nn.Linear(DINOv2RGBEncoder.FEATURE_DIM, image_dim), nn.GELU(),
            )
            self.point_fusion_head = nn.Sequential(
                nn.Linear(geometry_dim + image_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 3),
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(geometry_dim + image_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1),
            )
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.constant_(self.fusion_gate[-1].bias, math.log(self.initial_gate / (1 - self.initial_gate)))
        elif fusion_mode == "legacy":
            self.fusion_head = nn.Sequential(
                nn.Linear(geometry_dim + image_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, 3),
            )
        else:
            raise ValueError(f"unsupported fusion mode: {fusion_mode}")

    def _capture_geometry_features(self, module, inputs) -> None:
        self._geometry_features = inputs[0]

    def freeze_geometry(self) -> None:
        self.geometry_frozen = True
        for parameter in self.geometry.parameters():
            parameter.requires_grad_(False)
        self.geometry.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.geometry_frozen:
            self.geometry.eval()
        return self

    @staticmethod
    def _sample_dino_patch_tokens(patch_tokens: torch.Tensor, point_uv: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Bilinearly sample each point's DINO token map at its projected crop coordinate."""
        batch_size, token_count, feature_dim = patch_tokens.shape
        patch_side = int(math.isqrt(token_count))
        if patch_side * patch_side != token_count:
            raise ValueError(f"DINO patch token count {token_count} is not square")
        if point_uv.shape != (int(counts.sum()), 2):
            raise ValueError("point_uv must have one normalized crop coordinate per point")
        patch_maps = patch_tokens.transpose(1, 2).reshape(batch_size, feature_dim, patch_side, patch_side)
        point_batch = torch.repeat_interleave(torch.arange(batch_size, device=patch_tokens.device), counts.to(patch_tokens.device))
        grids = point_uv.to(dtype=patch_tokens.dtype).reshape(-1, 1, 1, 2)
        values = F.grid_sample(
            patch_maps.index_select(0, point_batch), grids, mode="bilinear", padding_mode="border", align_corners=False,
        )
        return values[:, :, 0, 0]

    def _limited_correction(self, raw_correction: torch.Tensor) -> torch.Tensor:
        if self.max_rgb_correction is None:
            return raw_correction
        correction_norm = raw_correction.norm(dim=1, keepdim=True).clamp_min(EPS)
        return raw_correction / correction_norm * (self.max_rgb_correction * torch.tanh(correction_norm))

    def forward(self, coord, radial, offset, image, counts, radial_weight_beta: float, rgb_dropout: float = 0.0,
                point_uv: torch.Tensor | None = None):
        if self.geometry_mode == "feature_head":
            point_features = self.geometry(coord, radial, offset)
            point_vectors = self.point_head(point_features)
            geometry_vector = radial_weighted_pool(point_vectors, counts, radial, radial_weight_beta)
        else:
            self._geometry_features = None
            point_vectors = self.geometry(coord, radial, offset)
            point_features = self._geometry_features
            if point_features is None:
                raise RuntimeError("MSECNet classifier hook did not capture geometry features")
            geometry_vector = radial_weighted_pool(F.normalize(point_vectors, dim=1, eps=EPS), counts, radial, radial_weight_beta)
        geometry_feature = radial_weighted_pool(point_features, counts, radial, radial_weight_beta)
        if self.fusion_mode == "gated_residual":
            rgb_feature = self.rgb(image)
            if self.training and rgb_dropout:
                keep = (torch.rand(len(rgb_feature), 1, device=rgb_feature.device) >= rgb_dropout).to(rgb_feature.dtype)
                rgb_feature = rgb_feature * keep
            fusion_input = torch.cat((geometry_feature, rgb_feature), dim=1)
            gate = torch.sigmoid(self.fusion_gate(fusion_input))
            correction = self._limited_correction(self.fusion_head(fusion_input))
            fused_vector = geometry_vector + gate * correction
        elif self.fusion_mode == "point_aligned_residual":
            if point_uv is None:
                raise ValueError("point_aligned_residual requires projected point_uv coordinates")
            rgb_feature, patch_tokens = self.rgb.forward_with_patch_tokens(image)
            point_visual = self.point_visual_adapter(self._sample_dino_patch_tokens(patch_tokens, point_uv, counts))
            if self.training and rgb_dropout:
                keep = (torch.rand(len(rgb_feature), 1, device=rgb_feature.device) >= rgb_dropout).to(rgb_feature.dtype)
                rgb_feature = rgb_feature * keep
                point_visual = point_visual * torch.repeat_interleave(keep, counts.to(keep.device), dim=0)
            visual_feature = radial_weighted_pool(point_visual, counts, radial, radial_weight_beta)
            fusion_input = torch.cat((geometry_feature, visual_feature), dim=1)
            gate = torch.sigmoid(self.fusion_gate(fusion_input))
            point_corrections = self.point_fusion_head(torch.cat((point_features, point_visual), dim=1))
            correction = self._limited_correction(radial_weighted_pool(point_corrections, counts, radial, radial_weight_beta))
            fused_vector = geometry_vector + gate * correction
        else:
            gate = None
            rgb_feature = self.rgb(image)
            if self.training and rgb_dropout:
                keep = (torch.rand(len(rgb_feature), 1, device=rgb_feature.device) >= rgb_dropout).to(rgb_feature.dtype)
                rgb_feature = rgb_feature * keep
            fusion_input = torch.cat((geometry_feature, rgb_feature), dim=1)
            fused_vector = self.fusion_head(fusion_input)
        return {
            "point_vectors": point_vectors,
            "geometry_vector": geometry_vector,
            "image_vector": self.image_head(rgb_feature),
            "fused_vector": fused_vector,
            "fusion_gate": gate if gate is not None else torch.ones((len(rgb_feature), 1), device=rgb_feature.device),
        }


def normalized(vector: torch.Tensor) -> torch.Tensor:
    return F.normalize(vector, dim=1, eps=EPS)
