#!/usr/bin/env python3
"""Review and repair normal labels after MSECNet inference.

This app keeps the normal target in both prepared-dataset representations in
sync: ``labels_manual3d.npz`` is used by training, while
``anchors_manual3d.json`` retains the editable 3D annotation and its pose.
``normal_fixes.json`` records the original and current values of each
correction with its review context.

Run from the project root with the point2normal environment:
    conda run --no-capture-output -n point2normal python web_label/app_fix_normal.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import struct
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_ROOT = ROOT / "data"
ACTIVE_SOURCE_DATASET = "fuelcap_pass_20260803_10847"
DEFAULT_THRESHOLD_DEG = 5.0
DISPLAY_BALL_RADIUS_M = 0.08
STATE_LOCK = threading.RLock()

JOB = {"state": "idle", "logs": []}
STATE = {
    "dataset": None,
    "checkpoint": None,
    "model_type": None,
    "split": None,
    "report_path": None,
    "items": [],
    "by_file": {},
    "base_normals": {},
    "source_assets": {},
    "only_above": True,
    "threshold_deg": DEFAULT_THRESHOLD_DEG,
}
RGB_OBB_REQUIREMENTS = {}

app = Flask(__name__, static_folder=None)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def flip_yz(vector) -> list[float]:
    """Convert between camera coordinates and the three.js display frame."""
    return [float(vector[0]), -float(vector[1]), -float(vector[2])]


def normalized(vector) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("normal must contain three finite values")
    length = float(np.linalg.norm(result))
    if length < 1e-8:
        raise ValueError("normal must not be zero")
    return result / length


def angular_error_deg(prediction, target, sign_invariant: bool) -> float:
    """Return a directed or axis-only angle in degrees."""
    dot = float(np.dot(normalized(prediction), normalized(target)))
    if sign_invariant:
        dot = abs(dot)
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def error_metric(model_type: str) -> str:
    return "axis_error_deg" if model_type == "pseudo_obb" else "angular_error_deg"


def is_sign_invariant(model_type: str) -> bool:
    return model_type == "pseudo_obb"


def original_labels_path(dataset_dir: Path) -> Path:
    return dataset_dir / "labels_manual3d.npz"


def fixes_path(dataset_dir: Path) -> Path:
    return dataset_dir / "normal_fixes.json"


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value) -> None:
    atomic_write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temp_name, **arrays)
        with open(temp_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_label_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"files", "normal"}
        if not required.issubset(archive.files):
            raise ValueError(f"{path} is missing {sorted(required - set(archive.files))}")
        return {name: archive[name].copy() for name in archive.files}


def load_normal_map(path: Path) -> dict[str, np.ndarray]:
    archive = load_label_archive(path)
    return {str(file_name): normalized(normal) for file_name, normal in zip(archive["files"], archive["normal"])}


def updated_anchor(anchor: dict, corrected_normal: np.ndarray) -> dict:
    """Return an anchor whose normal, tangent, and pose agree exactly.

    ``pose_T`` stores a right-handed frame as [tangent, normal x tangent,
    normal].  Merely replacing ``anchor[\"normal\"]`` leaves a stale pose and
    makes the prepared data invalid when it is regenerated, so rebuild the
    frame while retaining the old tangent direction as closely as possible.
    """
    if not isinstance(anchor, dict):
        raise ValueError("anchor must be an object")
    try:
        center = np.asarray(anchor["center_3d"], dtype=np.float64)
        tangent_hint = np.asarray(anchor["tangent"], dtype=np.float64)
        pose = np.asarray(anchor["pose_T"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("anchor is missing center_3d, tangent, or pose_T") from exc
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("anchor center_3d must contain three finite values")
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("anchor pose_T must be a finite 4x4 matrix")

    normal = normalized(corrected_normal)
    if np.linalg.norm(center) >= 1e-8 and float(np.dot(normal, -center)) <= 0:
        raise ValueError("corrected normal must face the camera")
    # Preserve the roll convention of the manual rectangle.  If the stored
    # tangent is degenerate after projection, use the old pose x-axis and then
    # a deterministic world axis as a last resort.
    for candidate in (tangent_hint, pose[:3, 0], np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0))):
        tangent = candidate - normal * float(np.dot(candidate, normal))
        if np.linalg.norm(tangent) >= 1e-8:
            tangent = normalized(tangent)
            break
    else:  # pragma: no cover - a unit normal always has an orthogonal world axis
        raise ValueError("cannot construct a tangent for corrected normal")
    bitangent = normalized(np.cross(normal, tangent))

    result = dict(anchor)
    result["normal"] = normal.tolist()
    result["tangent"] = tangent.tolist()
    pose = pose.copy()
    pose[:3, 0] = tangent
    pose[:3, 1] = bitangent
    pose[:3, 2] = normal
    pose[:3, 3] = center
    pose[3] = (0.0, 0.0, 0.0, 1.0)
    result["pose_T"] = pose.tolist()
    return result


def prepared_datasets() -> list[dict]:
    """Find reviewed prepared datasets and their compatible model families."""
    if not DATA_ROOT.is_dir():
        return []
    datasets = []
    for dataset_dir in sorted(DATA_ROOT.iterdir()):
        required = ("labels_manual3d.npz", "clouds", "anchors_manual3d.json", "manifest.jsonl")
        if not dataset_dir.is_dir() or not all((dataset_dir / name).exists() for name in required):
            continue
        split_path = None
        model_types = []
        if (dataset_dir / "split_by_generalization_group.json").is_file():
            split_path = dataset_dir / "split_by_generalization_group.json"
            model_types = ["center_ball", "pointnet2_ball", "rgb_ball"]
        elif (dataset_dir / "split_by_car_model.json").is_file():
            split_path = dataset_dir / "split_by_car_model.json"
            model_types = ["pseudo_obb"]
        if split_path is None:
            continue
        try:
            split = json.loads(split_path.read_text(encoding="utf-8"))
            manifest = [json.loads(line) for line in (dataset_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
        except (OSError, json.JSONDecodeError):
            continue
        # Test is intentionally available for diagnosis of reported long-tail
        # cases. Saving there changes the evaluation reference, so the UI marks
        # the split and keeps it in the per-sample audit record.
        splits = [name for name in ("train", "val", "test") if split.get(name)]
        if not manifest or not splits:
            continue
        source_dataset = ""
        metadata_path = dataset_dir / "dataset.json"
        if metadata_path.is_file():
            try:
                source_dataset = str(json.loads(metadata_path.read_text(encoding="utf-8")).get("source_dataset", ""))
            except json.JSONDecodeError:
                pass
        if source_dataset != ACTIVE_SOURCE_DATASET:
            continue
        source_root = DATA_ROOT / source_dataset if source_dataset else None
        source_cloud = manifest[0].get("source_cloud", "")
        if not source_root or not (source_root / source_cloud).is_file():
            source_root = next(
                (candidate for candidate in sorted(DATA_ROOT.glob("fuelcap_pass_*"))
                 if (candidate / source_cloud).is_file()),
                None,
            )
        datasets.append({
            "id": dataset_dir.name,
            "label": dataset_dir.name,
            "model_types": model_types,
            "splits": splits,
            "split_path": str(split_path),
            "source_root": str(source_root) if source_root else "",
        })
    return datasets


def available_checkpoints() -> list[dict]:
    def radius_label(path: Path, fallback: str) -> str:
        metadata_path = path.parent / "run.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            radius = float(metadata.get("ball_radius_m"))
            if radius > 0:
                return f"{radius * 100:g} cm 球形点云"
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return fallback

    checkpoints = []
    specs = (
        ("msecnet_ball", "center_ball", "8 cm 球形点云"),
        ("pointnet2_ball", "pointnet2_ball", "PointNet++ 球形点云"),
        ("msecnet_ball_addRGB", "rgb_ball", "RGB + 球形点云"),
        ("msecnet_best", "pseudo_obb", "人工伪 OBB"),
    )
    for directory, model_type, prefix in specs:
        for path in sorted((ROOT / directory / "out").glob("*/best.pt")):
            display_prefix = prefix
            if model_type == "pointnet2_ball":
                display_prefix = f"PointNet++ {radius_label(path, prefix)}"
            checkpoints.append({
                "id": str(path.relative_to(ROOT)),
                "label": f"{display_prefix}: {path.parent.name}",
                "model_type": model_type,
            })
    return checkpoints


def available_options() -> dict:
    checkpoints = available_checkpoints()
    available_types = {item["model_type"] for item in checkpoints}
    datasets = [item for item in prepared_datasets() if set(item["model_types"]) & available_types]
    return {"checkpoints": checkpoints, "datasets": datasets}


def find_option(options: dict, checkpoint_id: str, dataset_id: str) -> tuple[dict, dict]:
    checkpoint_by_id = {item["id"]: item for item in options["checkpoints"]}
    dataset_by_id = {item["id"]: item for item in options["datasets"]}
    if checkpoint_id not in checkpoint_by_id or dataset_id not in dataset_by_id:
        raise ValueError("未知的权重或准备数据集")
    checkpoint, dataset = checkpoint_by_id[checkpoint_id], dataset_by_id[dataset_id]
    if checkpoint["model_type"] not in dataset["model_types"]:
        raise ValueError("权重与准备数据集的输入定义不匹配")
    return checkpoint, dataset


def load_anchors(dataset_dir: Path) -> dict:
    path = dataset_dir / "anchors_manual3d.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def safe_source_path(source_root: Path, relative_path: str) -> Path | None:
    """Resolve a manifest path while keeping file serving inside the source dataset."""
    if not source_root.is_dir() or not relative_path:
        return None
    try:
        path = (source_root / relative_path).resolve()
        path.relative_to(source_root.resolve())
    except (OSError, ValueError):
        return None
    return path if path.is_file() else None


def source_assets_for_dataset(dataset: dict) -> dict[str, dict]:
    """Map prepared sample names to their original full cloud and RGB image."""
    dataset_dir = DATA_ROOT / dataset["id"]
    source_root = Path(dataset["source_root"])
    index_images = {}
    index_path = source_root / "index.jsonl"
    if index_path.is_file():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            try:
                row = json.loads(line)
                index_images[f"{row['id']}.npz"] = row.get("image", "")
            except (KeyError, json.JSONDecodeError):
                continue
    assets = {}
    manifest_path = dataset_dir / "manifest.jsonl"
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            row = json.loads(line)
            file_name = str(row["file"])
        except (KeyError, json.JSONDecodeError):
            continue
        assets[file_name] = {
            "source_cloud": str(row.get("source_cloud", "")),
            "source_label": str(row.get("source_label", "")),
            "source_image": str(index_images.get(file_name, "")),
        }
    return assets


def source_image_path(dataset: dict, asset: dict | None) -> Path | None:
    if not asset:
        return None
    source_root = Path(dataset["source_root"])
    image_path = safe_source_path(source_root, asset.get("source_image", ""))
    if image_path:
        return image_path
    label_path = safe_source_path(source_root, asset.get("source_label", ""))
    if not label_path:
        return None
    try:
        label = json.loads(label_path.read_text(encoding="utf-8"))
        return safe_source_path(source_root, label.get("meta", {}).get("source_image", ""))
    except (OSError, json.JSONDecodeError):
        return None


def update_repaired_label(
    dataset_dir: Path,
    file_name: str,
    corrected_normal: np.ndarray,
    review: dict,
) -> int:
    """Synchronize NPZ, anchor JSON, and audit data for one reviewed sample.

    Every individual file is written via atomic replacement.  All validation
    and serialization happens before replacing either source of truth, so a
    normal UI save cannot leave a half-built anchor frame behind.
    """
    labels_source = original_labels_path(dataset_dir)
    arrays = load_label_archive(labels_source)
    names = [str(value) for value in arrays["files"]]
    try:
        index = names.index(file_name)
    except ValueError as exc:
        raise ValueError(f"{file_name} is absent from {labels_source.name}") from exc
    normals = arrays["normal"].copy()
    normals[index] = corrected_normal.astype(normals.dtype, copy=False)
    arrays["normal"] = normals

    anchors_path = dataset_dir / "anchors_manual3d.json"
    anchors = load_anchors(dataset_dir)
    if file_name not in anchors:
        raise ValueError(f"{file_name} is absent from {anchors_path.name}")
    anchors[file_name] = updated_anchor(anchors[file_name], corrected_normal)

    audit_file = fixes_path(dataset_dir)
    if audit_file.is_file():
        try:
            audit = json.loads(audit_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            audit = {}
    else:
        audit = {}
    fixes = audit.setdefault("fixes", {})
    audit["schema_version"] = 1
    audit["dataset"] = dataset_dir.name
    audit["updated_at"] = review["saved_at"]
    previous = fixes.get(file_name)
    if previous:
        history = list(previous.get("history", []))
        previous_record = dict(previous)
        previous_record.pop("history", None)
        history.append(previous_record)
        review = dict(review)
        review["history"] = history[-50:]
    fixes[file_name] = review
    fixes[file_name]["anchors_synchronized"] = True

    # Training consumes the NPZ; regeneration and visual inspection consume
    # anchors.  Update both on the same Save action, then append its audit
    # entry only after the two canonical representations are in sync.
    atomic_write_npz(labels_source, arrays)
    atomic_write_json(anchors_path, anchors)
    atomic_write_json(audit_file, audit)
    return len(fixes)


def report_rows(report_path: Path) -> list[dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("predictions") if isinstance(report, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{report_path} 中没有推理结果")
    if any(not isinstance(row, dict) or "file" not in row or "pred_normal" not in row for row in rows):
        raise ValueError(f"{report_path} 的推理结果不完整")
    return rows


def build_review_items(report_path: Path, dataset_dir: Path, model_type: str, only_above: bool, threshold_deg: float) -> tuple[list[dict], dict]:
    if threshold_deg < 0 or not np.isfinite(threshold_deg):
        raise ValueError("角度阈值必须是非负有限数")
    current_normals = load_normal_map(original_labels_path(dataset_dir))
    base_normals = dict(current_normals)
    audit_file = fixes_path(dataset_dir)
    if audit_file.is_file():
        try:
            audit_fixes = json.loads(audit_file.read_text(encoding="utf-8")).get("fixes", {})
            for file_name, record in audit_fixes.items():
                if file_name in base_normals and "base_normal" in record:
                    base_normals[file_name] = normalized(record["base_normal"])
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    metric = error_metric(model_type)
    sign_invariant = is_sign_invariant(model_type)
    items = []
    for row in report_rows(report_path):
        file_name = str(row["file"])
        if file_name not in current_normals or file_name not in base_normals:
            continue
        prediction = normalized(row["pred_normal"])
        current = current_normals[file_name]
        error = angular_error_deg(prediction, current, sign_invariant)
        if only_above and not error > threshold_deg:
            continue
        items.append({
            "file": file_name,
            "prediction": prediction.tolist(),
            "target": current.tolist(),
            "base_normal": base_normals[file_name].tolist(),
            "error_deg": error,
            "row": row,
        })
    items.sort(key=lambda item: (-item["error_deg"], item["file"]))
    return items, base_normals


def set_review_queue(
    report_path: Path,
    dataset: dict,
    checkpoint: dict,
    split_name: str,
    only_above: bool,
    threshold_deg: float,
) -> dict:
    dataset_dir = DATA_ROOT / dataset["id"]
    items, base_normals = build_review_items(
        report_path, dataset_dir, checkpoint["model_type"], only_above, threshold_deg
    )
    source_assets = source_assets_for_dataset(dataset)
    with STATE_LOCK:
        STATE.update({
            "dataset": dataset,
            "checkpoint": checkpoint,
            "model_type": checkpoint["model_type"],
            "split": split_name,
            "report_path": str(report_path),
            "items": items,
            "by_file": {item["file"]: item for item in items},
            "base_normals": base_normals,
            "source_assets": source_assets,
            "only_above": only_above,
            "threshold_deg": threshold_deg,
        })
    return queue_meta()


def queue_meta() -> dict:
    with STATE_LOCK:
        dataset = STATE["dataset"]
        checkpoint = STATE["checkpoint"]
        return {
            "n": len(STATE["items"]),
            "dataset": dataset["id"] if dataset else None,
            "checkpoint": checkpoint["id"] if checkpoint else None,
            "model_type": STATE["model_type"],
            "split": STATE["split"],
            "report_path": STATE["report_path"],
            "only_above": STATE["only_above"],
            "threshold_deg": STATE["threshold_deg"],
            "labels_path": str(original_labels_path(DATA_ROOT / dataset["id"])) if dataset else None,
        }


def append_job_log(message: str) -> None:
    with STATE_LOCK:
        logs = JOB.setdefault("logs", [])
        logs.append(str(message).rstrip()[:1200])
        del logs[:-200]


def current_job() -> dict:
    with STATE_LOCK:
        result = dict(JOB)
        result["logs"] = list(JOB.get("logs", [])[-80:])
        return result


def rgb_checkpoint_requires_obb(checkpoint_path: Path) -> bool:
    cached = RGB_OBB_REQUIREMENTS.get(checkpoint_path)
    if cached is not None:
        return cached
    # Torch is already required by inference.  Load on CPU only so the web
    # process can determine the exact image-crop protocol before starting GPU work.
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = checkpoint.get("input_schema", {}).get("rgb_crop_source") == "detector_obb"
    RGB_OBB_REQUIREMENTS[checkpoint_path] = required
    return required


def inference_command(checkpoint: dict, dataset: dict, split_name: str) -> tuple[list[str], Path, Path]:
    checkpoint_path = ROOT / checkpoint["id"]
    dataset_dir = DATA_ROOT / dataset["id"]
    labels_path = original_labels_path(dataset_dir)
    anchors_path = dataset_dir / "anchors_manual3d.json"
    split_path = Path(dataset["split_path"])
    output_dir = checkpoint_path.parent / "repair_inference" / dataset_dir.name / split_name
    base = [sys.executable, "-u"]
    if checkpoint["model_type"] == "center_ball":
        command = base + [
            str(ROOT / "msecnet_ball" / "infer.py"), str(checkpoint_path), str(labels_path), str(dataset_dir / "clouds"),
            "--centers", str(anchors_path), "--split", str(split_path), "--split-name", split_name,
            "--out", str(output_dir),
        ]
    elif checkpoint["model_type"] == "pointnet2_ball":
        command = base + [
            str(ROOT / "pointnet2_ball" / "infer.py"), str(checkpoint_path), str(labels_path), str(dataset_dir / "clouds"),
            "--centers", str(anchors_path), "--split", str(split_path), "--split-name", split_name,
            "--out", str(output_dir),
        ]
    elif checkpoint["model_type"] == "pseudo_obb":
        command = base + [
            str(ROOT / "msecnet_best" / "infer.py"), str(checkpoint_path), str(labels_path), str(dataset_dir / "clouds"),
            "--centers", str(anchors_path), "--split", str(split_path), "--split-name", split_name,
            "--out", str(output_dir),
        ]
    elif checkpoint["model_type"] == "rgb_ball":
        source_root = Path(dataset["source_root"])
        if not source_root.is_dir():
            raise ValueError("RGB 融合推理需要可用的原始数据目录")
        command = base + [
            str(ROOT / "msecnet_ball_addRGB" / "infer.py"), str(checkpoint_path), str(labels_path),
            str(dataset_dir / "clouds"), "--source-root", str(source_root), "--centers", str(anchors_path),
            "--split", str(split_path), "--split-name", split_name, "--out", str(output_dir),
        ]
        if rgb_checkpoint_requires_obb(checkpoint_path):
            candidates = sorted(dataset_dir.glob("obb*.json"))
            if not candidates:
                raise ValueError(
                    "该 RGB 权重要求 detector OBB 缓存；请先在准备数据集目录生成 obb*.json，再启动推理"
                )
            command.extend(["--obb-detections", str(candidates[-1])])
    else:
        raise ValueError("不支持的模型类型")
    return command, output_dir / "report.json", dataset_dir


def run_inference_job(
    command: list[str],
    report_path: Path,
    dataset: dict,
    checkpoint: dict,
    split_name: str,
    only_above: bool,
    threshold_deg: float,
) -> None:
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        with STATE_LOCK:
            JOB["pid"] = process.pid
        assert process.stdout is not None
        for line in process.stdout:
            append_job_log(line)
        if process.wait() != 0:
            raise RuntimeError(f"infer.py 以状态码 {process.returncode} 退出")
        meta = set_review_queue(report_path, dataset, checkpoint, split_name, only_above, threshold_deg)
        with STATE_LOCK:
            JOB.update({"state": "complete", "summary": {"review_samples": meta["n"]}, "pid": None})
        append_job_log("推理完成，已加载修复队列。")
    except Exception as exc:
        append_job_log(f"ERROR: {exc}")
        with STATE_LOCK:
            JOB.update({"state": "failed", "error": str(exc), "pid": None})


def cloud_response(item: dict, index: int) -> Response:
    with STATE_LOCK:
        dataset = STATE["dataset"]
        model_type = STATE["model_type"]
        split_name = STATE["split"]
        asset = STATE["source_assets"].get(item["file"])
        queue_length = len(STATE["items"])
    if not dataset or not model_type:
        abort(404, "尚未加载推理结果")
    dataset_dir = DATA_ROOT / dataset["id"]
    cloud_path = dataset_dir / "clouds" / item["file"]
    if not cloud_path.is_file():
        abort(404, f"找不到点云: {item['file']}")
    display_mode = request.args.get("mode", "input")
    with np.load(cloud_path, allow_pickle=False) as archive:
        input_xyz = archive["xyz"].astype(np.float32)
        input_label = archive["label"] if "label" in archive else np.ones(len(input_xyz), dtype=np.int8)
        input_rgb = archive["rgb"].astype(np.float32) if "rgb" in archive else None
    anchors = load_anchors(dataset_dir)
    anchor = anchors.get(item["file"], {})
    full_cloud_path = safe_source_path(Path(dataset["source_root"]), asset.get("source_cloud", "") if asset else "")
    if display_mode == "full" and full_cloud_path and full_cloud_path.suffix.lower() == ".npz":
        with np.load(full_cloud_path, allow_pickle=False) as archive:
            xyz = archive["xyz"].astype(np.float32)
            rgb = archive["rgb"].astype(np.float32) if "rgb" in archive else None
        center = np.asarray(anchor.get("center_3d", np.median(xyz, axis=0)), dtype=np.float32)
        scale = float(np.percentile(np.linalg.norm(xyz - center, axis=1), 97) + 1e-8)
        geometry_name = "全局原始点云"
    else:
        display_mode = "input"
        point_mask = input_label == 1
        xyz = input_xyz[point_mask] if point_mask.any() else input_xyz
        rgb = input_rgb[point_mask] if input_rgb is not None and point_mask.any() else input_rgb
        if model_type in ("center_ball", "pointnet2_ball", "rgb_ball") and "center_3d" in anchor:
            center = np.asarray(anchor["center_3d"], dtype=np.float32)
            radius = float(anchor.get("ball_radius_m", DISPLAY_BALL_RADIUS_M))
            ball_mask = np.linalg.norm(xyz - center, axis=1) <= radius
            if ball_mask.any():
                xyz = xyz[ball_mask]
                if rgb is not None:
                    rgb = rgb[ball_mask]
            scale = radius
            geometry_name = f"{radius * 100:.0f} cm 球形模型输入"
        else:
            center = np.asarray(anchor.get("center_3d", np.median(xyz, axis=0)), dtype=np.float32)
            scale = float(np.percentile(np.linalg.norm(xyz - center, axis=1), 97) + 1e-8)
            geometry_name = "人工伪 OBB 模型输入"
    if len(xyz) == 0:
        abort(422, f"{item['file']} 没有可显示的输入点")
    if len(xyz) > 160000:
        selected = np.random.default_rng(0).choice(len(xyz), 160000, replace=False)
        xyz = xyz[selected]
        if rgb is not None:
            rgb = rgb[selected]
    if rgb is None:
        rgb = np.tile(np.asarray((0.55, 0.68, 0.85), dtype=np.float32), (len(xyz), 1))
    elif rgb.max() > 1.0:
        rgb = rgb / 255.0
    low, high = np.percentile(rgb, 3, axis=0), np.percentile(rgb, 97, axis=0)
    rgb = np.clip((rgb - low) / (high - low + 1e-6), 0, 1)
    points = ((xyz - center) / scale) * np.asarray((1, -1, -1), dtype=np.float32)
    prediction = normalized(item["prediction"])
    target = normalized(item["target"])
    if is_sign_invariant(model_type) and float(np.dot(prediction, target)) < 0:
        prediction = -prediction
    base_normal = normalized(item["base_normal"])
    header = {
        "i": index,
        "n": queue_length,
        "file": item["file"],
        "normal": flip_yz(target),
        "base_normal": flip_yz(base_normal),
        "model_label": flip_yz(prediction),
        "error_deg": item["error_deg"],
        "metric": error_metric(model_type),
        "geometry": geometry_name,
        "display_mode": display_mode,
        "npts": int(len(points)),
        "has_rgb": source_image_path(dataset, asset) is not None,
        "is_fixed": not np.allclose(target, base_normal, atol=1e-7),
        "model_type": model_type,
        "car_model": item.get("row", {}).get("car_model", ""),
        "split": split_name or "",
    }
    packed = np.empty((len(points), 6), dtype=np.float32)
    packed[:, :3] = points
    packed[:, 3:] = rgb
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    header_bytes += b" " * ((-(4 + len(header_bytes))) % 4)
    return Response(
        struct.pack("<I", len(header_bytes)) + header_bytes + packed.tobytes(),
        mimetype="application/octet-stream",
    )


@app.route("/")
def index():
    return send_from_directory(HERE, "fix_normal.html")


@app.route("/vendor/<path:asset>")
def vendor(asset):
    return send_from_directory(HERE / "vendor", asset)


@app.route("/api/options")
def options():
    return jsonify(available_options())


@app.route("/api/job")
def job_status():
    return jsonify(current_job())


@app.route("/api/job", methods=["POST"])
def start_job():
    payload = request.get_json(silent=True) or {}
    checkpoint_id = payload.get("checkpoint")
    dataset_id = payload.get("dataset")
    split_name = payload.get("split")
    only_above = bool(payload.get("only_above", True))
    try:
        threshold_deg = float(payload.get("threshold_deg", DEFAULT_THRESHOLD_DEG))
        checkpoint, dataset = find_option(available_options(), checkpoint_id, dataset_id)
        if split_name not in dataset["splits"] or split_name not in ("train", "val", "test"):
            raise ValueError("无效的划分；请选择 train、val 或 test")
        if threshold_deg < 0 or not np.isfinite(threshold_deg):
            raise ValueError("角度阈值必须是非负有限数")
        command, report_path, _ = inference_command(checkpoint, dataset, split_name)
    except (TypeError, ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400
    with STATE_LOCK:
        if JOB.get("state") == "running":
            return jsonify({"error": "已有推理任务正在运行"}), 409
        JOB.clear()
        JOB.update({
            "state": "running",
            "checkpoint": checkpoint_id,
            "dataset": dataset_id,
            "split": split_name,
            "logs": ["正在启动推理..."],
            "output_dir": str(report_path.parent),
        })
    threading.Thread(
        target=run_inference_job,
        args=(command, report_path, dataset, checkpoint, split_name, only_above, threshold_deg),
        daemon=True,
    ).start()
    return jsonify(current_job()), 202


@app.route("/api/queue", methods=["POST"])
def rebuild_queue():
    payload = request.get_json(silent=True) or {}
    try:
        only_above = bool(payload.get("only_above", True))
        threshold_deg = float(payload.get("threshold_deg", DEFAULT_THRESHOLD_DEG))
        if threshold_deg < 0 or not np.isfinite(threshold_deg):
            raise ValueError("角度阈值必须是非负有限数")
        with STATE_LOCK:
            report_path = STATE["report_path"]
            dataset = STATE["dataset"]
            checkpoint = STATE["checkpoint"]
            split_name = STATE["split"]
        if not report_path or not dataset or not checkpoint:
            raise ValueError("请先完成一次推理")
        return jsonify(set_review_queue(
            Path(report_path), dataset, checkpoint, split_name, only_above, threshold_deg
        ))
    except (TypeError, ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/meta")
def meta():
    return jsonify(queue_meta())


@app.route("/api/cloud/<int:index>")
def cloud(index: int):
    with STATE_LOCK:
        if index < 0 or index >= len(STATE["items"]):
            abort(404, "没有这个修复样本")
        item = dict(STATE["items"][index])
    return cloud_response(item, index)


@app.route("/api/rgb/<int:index>")
def rgb_image(index: int):
    with STATE_LOCK:
        if index < 0 or index >= len(STATE["items"]):
            abort(404, "没有这个修复样本")
        dataset = STATE["dataset"]
        asset = STATE["source_assets"].get(STATE["items"][index]["file"])
    image_path = source_image_path(dataset, asset) if dataset else None
    if not image_path:
        abort(404, "当前样本没有源 RGB 图")
    return send_file(image_path, conditional=True)


@app.route("/api/save", methods=["POST"])
def save():
    payload = request.get_json(silent=True) or {}
    try:
        file_name = str(payload["file"])
        displayed_normal = normalized(payload["normal"])
        with STATE_LOCK:
            dataset = STATE["dataset"]
            checkpoint = STATE["checkpoint"]
            model_type = STATE["model_type"]
            split_name = STATE["split"]
            report_path = STATE["report_path"]
            item = STATE["by_file"].get(file_name)
        if not dataset or not checkpoint or not item:
            raise ValueError("样本不在当前修复队列中")
        corrected = normalized(flip_yz(displayed_normal))
        previous = normalized(item["target"])
        if is_sign_invariant(model_type) and float(np.dot(corrected, previous)) < 0:
            corrected = -corrected
        prediction = normalized(item["prediction"])
        current_error = angular_error_deg(prediction, corrected, is_sign_invariant(model_type))
        base_normal = normalized(item["base_normal"])
        review = {
            "base_normal": base_normal.tolist(),
            "previous_normal": previous.tolist(),
            "normal": corrected.tolist(),
            "prediction": prediction.tolist(),
            "prediction_error_deg": current_error,
            "metric": error_metric(model_type),
            "checkpoint": checkpoint["id"],
            "report": report_path,
            "split": split_name,
            "saved_at": utc_now(),
        }
        count = update_repaired_label(DATA_ROOT / dataset["id"], file_name, corrected, review)
        with STATE_LOCK:
            STATE["by_file"][file_name]["target"] = corrected.tolist()
            STATE["by_file"][file_name]["error_deg"] = current_error
            for queued in STATE["items"]:
                if queued["file"] == file_name:
                    queued["target"] = corrected.tolist()
                    queued["error_deg"] = current_error
                    break
        return jsonify({"ok": True, "fixed_count": count, "error_deg": current_error})
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


def main() -> None:
    parser = argparse.ArgumentParser(description="MSECNet normal-label repair web app")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    print(f"normal repair app: http://0.0.0.0:{args.port}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
