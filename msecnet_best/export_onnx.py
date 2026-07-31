#!/usr/bin/env python3
"""Export the legacy MSECNet normal estimator as a portable ONNX model.

MSECNet normally uses CUDA-only pointops extensions.  This exporter replaces
those operations with equivalent PyTorch tensor operations while tracing, so
the resulting ONNX graph has no custom operators.  The exported graph accepts
one already-preprocessed point cloud of a fixed size and produces per-point
raw normal vectors plus the normalized normal of the complete cloud.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
MROOT = HERE / "MSECNet"
sys.path.insert(0, str(MROOT / "model"))
sys.path.insert(0, str(MROOT / "scripts"))

from architectures import MSECNet  # noqa: E402
from lib.pointops.functions import pointops  # noqa: E402
from util import config  # noqa: E402


def _single_batch_offset(offset, new_offset):
    """Reject packed batches: ONNX export has one fixed-size cloud per call."""
    if offset.numel() != 1 or new_offset.numel() != 1:
        raise ValueError("the ONNX export path supports exactly one point cloud per invocation")


def onnx_knnquery(nsample, xyz, new_xyz, offset, new_offset):
    """ONNX-friendly exact KNN replacement for pointops.knnquery."""
    _single_batch_offset(offset, new_offset)
    if new_xyz is None:
        new_xyz = xyz
    # Avoid torch.cdist because it does not have a portable ONNX lowering.
    delta = new_xyz.unsqueeze(1) - xyz.unsqueeze(0)
    dist2 = (delta * delta).sum(dim=-1)
    k = min(nsample, xyz.shape[0])
    values, indices = torch.topk(dist2, k=k, dim=1, largest=False, sorted=True)
    if k < nsample:
        # Match the CUDA implementation, which repeats the first neighbour
        # when a cloud contains fewer than nsample points.
        pad = nsample - k
        indices = torch.cat((indices, indices[:, :1].expand(-1, pad)), dim=1)
        values = torch.cat((values, values[:, :1].expand(-1, pad)), dim=1)
    return indices.to(torch.int32), torch.sqrt(values)


def onnx_furthestsampling(xyz, offset, new_offset):
    """Deterministic first-point FPS equivalent to the bundled CUDA kernel."""
    _single_batch_offset(offset, new_offset)
    sample_count = int(new_offset[0].item())
    if sample_count < 1 or sample_count > xyz.shape[0]:
        raise ValueError("invalid furthest-point sample count")
    selected = [torch.zeros((), dtype=torch.long, device=xyz.device)]
    minimum_dist2 = torch.full((xyz.shape[0],), float("inf"), device=xyz.device, dtype=xyz.dtype)
    last = xyz[0:1]
    for _ in range(1, sample_count):
        delta = xyz - last
        minimum_dist2 = torch.minimum(minimum_dist2, (delta * delta).sum(dim=1))
        # TensorRT requires ArgMax inputs to have rank >= 2.  The extra
        # leading dimension is removed immediately so the FPS index remains
        # the same scalar used by the CUDA implementation.
        next_index = torch.argmax(minimum_dist2.unsqueeze(0), dim=1).squeeze(0)
        selected.append(next_index)
        last = xyz[next_index:next_index + 1]
    return torch.stack(selected).to(torch.int32)


def _onnx_interpolate(xyz, new_xyz, feat, offset, new_offset, k=3, weight_type="spatial"):
    idx, dist = onnx_knnquery(k, xyz, new_xyz, offset, new_offset)
    if weight_type == "spatial":
        reciprocal = 1.0 / (dist + 1e-8)
        weights = reciprocal / reciprocal.sum(dim=1, keepdim=True)
    elif weight_type == "gauss":
        weights = torch.exp(-(dist * dist) / ((0.3 ** 2) * 2 + 1e-8))
        weights = weights / weights.sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"unsupported interpolation weight type: {weight_type}")
    return (feat[idx.long()] * weights.unsqueeze(-1)).sum(dim=1)


def install_onnx_pointops_fallbacks():
    """Patch only this process so MSECNet traces with standard tensor ops."""
    def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset,
                      use_xyz=True, return_index=False):
        if new_xyz is None:
            new_xyz = xyz
        if idx is None:
            idx, _ = onnx_knnquery(nsample, xyz, new_xyz, offset, new_offset)
        idx_long = idx.long()
        grouped_xyz = xyz[idx_long] - new_xyz.unsqueeze(1)
        grouped_feat = feat[idx_long]
        result = torch.cat((grouped_xyz, grouped_feat), dim=-1) if use_xyz else grouped_feat
        return (result, idx_long) if return_index else result

    pointops.knnquery = onnx_knnquery
    pointops.furthestsampling = onnx_furthestsampling
    pointops.queryandgroup = queryandgroup
    pointops.interpolation = _onnx_interpolate
    pointops.interpolation_flexible = _onnx_interpolate


class MSECNetONNXWrapper(nn.Module):
    """Expose MSECNet through the fixed ``[1, N, 3]`` deployment interface."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, points):
        point_count = points.shape[1]
        coords = points[0]
        features = coords.new_zeros((point_count, 0))
        offset = torch.tensor([point_count], dtype=torch.int32, device=coords.device)
        raw_point_normals = self.model(coords, features, offset)
        cloud_normal = F.normalize(raw_point_normals.mean(dim=0), dim=0, eps=1e-12)
        return raw_point_normals.unsqueeze(0), cloud_normal.unsqueeze(0)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"{checkpoint_path} is missing model weights")
    cfg = config.load_cfg_from_cfg_file(str(MROOT / "scripts/config/pcpnet/config.yaml"))
    cfg.num_classes = 3
    model = MSECNet(cfg).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Export MSECNet to standard ONNX without CUDA pointops")
    parser.add_argument("checkpoint", type=Path, nargs="?", default=HERE / "checkpoints/best.pt")
    parser.add_argument("--out", type=Path, default=None, help="output .onnx path (default: next to checkpoint)")
    parser.add_argument("--num-points", type=int, default=None,
                        help="fixed point count; defaults to the checkpoint max_points value")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                        help="export device; CUDA is recommended because the source model is CUDA-oriented")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")

    model, checkpoint = load_model(args.checkpoint, args.device)
    num_points = args.num_points or int(checkpoint.get("max_points", checkpoint.get("npoints", 1024)))
    if num_points < 128 or num_points % 8:
        raise ValueError("--num-points must be at least 128 and divisible by 8 for the three downsampling stages")
    output_path = args.out or args.checkpoint.with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    install_onnx_pointops_fallbacks()
    wrapper = MSECNetONNXWrapper(model).to(args.device).eval()
    example = torch.zeros((1, num_points, 3), dtype=torch.float32, device=args.device)
    with torch.inference_mode():
        raw_point_normals, cloud_normal = wrapper(example)
    print(f"PyTorch outputs: point_normals={tuple(raw_point_normals.shape)} cloud_normal={tuple(cloud_normal.shape)}")
    torch.onnx.export(
        wrapper,
        (example,),
        str(output_path),
        input_names=("points",),
        output_names=("point_normals", "cloud_normal"),
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX export requires the 'onnx' package in the point2normal environment") from exc
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"exported={output_path.resolve()}")
    print(f"input=points[1,{num_points},3] outputs=point_normals[1,{num_points},3], cloud_normal[1,3]")


if __name__ == "__main__":
    main()
