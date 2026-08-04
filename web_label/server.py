#!/usr/bin/env python3
"""Browser-based cap-normal labeler (Flask + three.js/WebGL).

Serves the inner_cover sub-cloud + reconstructed source photo; saves corrected normals to
output/manual_normals.json.

Run:  python web_label/server.py [--port 8765]   then open http://<host>:8765
"""
import base64
import glob
import json
import os
import subprocess
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def arg_value(name, default=""):
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    if i + 1 >= len(sys.argv):
        return default
    return sys.argv[i + 1]


def load_prediction_dict(path):
    """Load either a batch prediction dict or one infer_pipeline output JSON."""
    if not path or not os.path.exists(path):
        return {}, None
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, dict) and "normal" in data and "cloud_npz" in data:
        key = os.path.basename(data["cloud_npz"])
        pred = {
            "normal": data["normal"],
            "points": data.get("points"),
            "point_source": data.get("point_source"),
            "checkpoint_step": data.get("checkpoint_step"),
            "checkpoint_mean_err": data.get("checkpoint_mean_err"),
            "source_json": path,
        }
        return {key: pred}, key
    return data, None

# ---- paths ----
PCD = os.path.join(ROOT, "data", "pcd_dataset_roi")
SAVE = os.path.join(ROOT, "output", "manual_normals.json")
LBL = os.path.join(ROOT, "shared", "normal_labels_full.npz")
KCJSON = os.path.join(ROOT, "shared", "knob_centers.json")
V3PATH = os.path.join(ROOT, "shared", "v3_predictions.json")
MODELPATH = (arg_value("--pred-json") or os.environ.get(
    "MSECNET_PRED_PATH", os.path.join(ROOT, "shared", "msecnet_predictions.json")
))
PRODPATH = os.environ.get("PROD_PRED_PATH", os.path.join(ROOT, "shared", "prod_predictions.json"))
MOGEPATH = os.environ.get("MOGE_PRED_PATH", os.path.join(ROOT, "shared", "moge_norm_predictions.json"))
AUDITPATH = os.path.join(ROOT, "shared", "data_audit.json")
FLAG = os.path.join(ROOT, "output", "flagged.json")
SRC_DIR = arg_value("--src-dir") or os.environ.get("SRC_DIR", os.path.join(ROOT, "data", "yolo_seg_by_car"))
EVAL_REPORT_PATH = arg_value("--msecnet-report")
EVAL_DATASET_DIR = arg_value("--msecnet-dataset")
MSECNET_UI_MODE = "--msecnet-ui" in sys.argv
EVAL_MODE = bool(EVAL_REPORT_PATH) or MSECNET_UI_MODE
EVAL_ROWS = {}
EVAL_ANCHORS = {}
EVAL_SOURCE_PATHS = {}
EVAL_ORIENTED = False
EVAL_HAS_TARGET = False
EVAL_BALL_RADIUS = None
DISPLAY_BALL_RADIUS_M = 0.08
EVAL_STATE_LOCK = threading.RLock()
INFERENCE_JOB = {"state": "idle", "logs": []}


def report_consensus(row):
    """Use bounded point agreement from new reports, with legacy fallback."""
    value = row.get("point_consensus")
    return float(value) if value is not None else float(row.get("mean_vector_norm", 0.0))


def report_error(row):
    return float(row.get("angular_error_deg", row["axis_error_deg"]))

os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)

# ---- load labels ----
L = None if EVAL_MODE else (np.load(LBL) if os.path.exists(LBL) else None)
if L is None and not EVAL_MODE:
    print("ERROR: normal_labels_full.npz not found. Generate labels first.", flush=True)
    sys.exit(1)

if L is None:
    FILES, NORMS, CONF, AGREE, NINNER = [], [], [], [], []
else:
    _files, _norm, _inl, _agr, _ni = L["files"], L["normal"], L["inlier_frac"], L["agree_deg"], L["n_inner"]
    order = np.argsort(-_ni)                               # CLEAREST (most inner points) first
    FILES = [str(_files[k]) for k in order]
    NORMS = [_norm[k].astype(float).tolist() for k in order]
    CONF = [float(_inl[k]) for k in order]
    AGREE = [float(_agr[k]) for k in order]
    NINNER = [int(_ni[k]) for k in order]

# v3 model-assisted pre-labels + AUTO-FILTER
V3 = json.load(open(V3PATH)) if os.path.exists(V3PATH) else {}
# MODEL prelabel (orange) shown vs geometric (red)
MODELPATH_USE = MODELPATH if os.path.exists(MODELPATH) else (PRODPATH if os.path.exists(PRODPATH) else None)
PROD, SINGLE_PRED_FILE = load_prediction_dict(MODELPATH_USE) if MODELPATH_USE else ({}, None)
if MODELPATH_USE:
    print(f"model prelabel source: {os.path.basename(MODELPATH_USE)} ({len(PROD)} preds)", flush=True)
MOGE = json.load(open(MOGEPATH)) if os.path.exists(MOGEPATH) else {}
if MOGE:
    print(f"moge-normal prelabel: {len(MOGE)} preds", flush=True)
_kc0 = json.load(open(KCJSON)) if os.path.exists(KCJSON) else {}
_AUDIT = json.load(open(AUDITPATH)) if os.path.exists(AUDITPATH) else {}


def _tier(f, i):
    probs = _AUDIT.get(f, {}).get("probs", [])
    if "detached" in probs:
        return "drop"
    kc = _kc0.get(f); v3 = V3.get(f)
    if kc is None or v3 is None:
        return "rest"
    asp = max(kc["wf"], kc["hf"]) / (min(kc["wf"], kc["hf"]) + 1e-6)
    if NINNER[i] < 600 or kc.get("conf", 1.0) < 0.55 or asp > 6:
        return "drop"
    if "dark" in probs or asp > 4 or v3.get("slant", 0) > 68:
        return "review"
    return "clean"


TIER = {}
if V3:
    TIER = {FILES[i]: _tier(FILES[i], i) for i in range(len(FILES))}
    rank = {"clean": 0, "review": 1, "rest": 2}
    inl_of = lambda i: (V3[FILES[i]]["inlier"] if FILES[i] in V3 else 0.0)  # noqa: E731
    _flag0 = json.load(open(FLAG)) if os.path.exists(FLAG) else {}
    keep = [i for i in range(len(FILES))
            if TIER[FILES[i]] != "drop" and FILES[i] not in _flag0]
    keep.sort(key=lambda i: (rank[TIER[FILES[i]]], -inl_of(i)))
    FILES = [FILES[i] for i in keep]; NORMS = [NORMS[i] for i in keep]
    CONF = [CONF[i] for i in keep]; AGREE = [AGREE[i] for i in keep]; NINNER = [NINNER[i] for i in keep]
    nb = {}
    for t in TIER.values():
        nb[t] = nb.get(t, 0) + 1
    print(f"v3 predictions: {len(V3)} | auto-filter: clean={nb.get('clean',0)} review={nb.get('review',0)} "
          f"rest={nb.get('rest',0)} DROPPED={nb.get('drop',0)} | labeling queue = {len(FILES)}", flush=True)
else:
    # RANSAC-based tiering — no model predictions needed.
    # agree_deg: cross-method disagreement (RANSAC vs MoGe).  P50=12.1° P75=18.5° P90=25.4°
    # inlier_frac: RANSAC self-confidence.  P25=0.41 P50=0.49  (inherently low due to surface curvature)
    def _tier_ransac(i):
        if NINNER[i] < 600:
            return "drop"
        if AGREE[i] > 20.0 or CONF[i] < 0.4:
            return "review"
        if AGREE[i] <= 8.0 and CONF[i] >= 0.7:
            return "clean"
        return "mid"

    TIER = {FILES[i]: _tier_ransac(i) for i in range(len(FILES))}
    rank = {"review": 0, "mid": 1, "clean": 2}          # most suspicious first
    _flag0 = json.load(open(FLAG)) if os.path.exists(FLAG) else {}
    keep = [i for i in range(len(FILES))
            if TIER[FILES[i]] != "drop" and FILES[i] not in _flag0]
    keep.sort(key=lambda i: (rank[TIER[FILES[i]]], -AGREE[i]))   # within tier: worst agree first
    FILES = [FILES[i] for i in keep]; NORMS = [NORMS[i] for i in keep]
    CONF = [CONF[i] for i in keep]; AGREE = [AGREE[i] for i in keep]; NINNER = [NINNER[i] for i in keep]
    nb = {}
    for t in TIER.values():
        nb[t] = nb.get(t, 0) + 1
    print(f"RANSAC auto-filter: review={nb.get('review',0)} mid={nb.get('mid',0)} "
          f"clean={nb.get('clean',0)} DROPPED={nb.get('drop',0)} | labeling queue = {len(FILES)}", flush=True)

FOCUS_FILE = "" if EVAL_MODE else (arg_value("--focus-file") or os.environ.get("FOCUS_FILE", "") or SINGLE_PRED_FILE or "").strip()
if FOCUS_FILE:
    focus_base = os.path.basename(FOCUS_FILE)
    if not focus_base.endswith(".npz"):
        focus_base += ".npz"
    focus = [i for i, f in enumerate(FILES) if f == focus_base]
    if focus:
        i = focus[0]
        FILES = [FILES[i]]
        NORMS = [NORMS[i]]
        CONF = [CONF[i]]
        AGREE = [AGREE[i]]
        NINNER = [NINNER[i]]
        print(f"FOCUS_FILE enabled: {focus_base}", flush=True)
    else:
        print(f"WARNING: FOCUS_FILE not found in labeling queue: {focus_base}", flush=True)

# Manual-3D MSECNet evaluation viewer. This keeps the original labeler mode unchanged.
def configure_evaluation(report_path, dataset_dir, source_root=None):
    """Load one completed infer.py report as the current read-only web view."""
    global PCD, FILES, NORMS, CONF, AGREE, NINNER, TIER, PROD, MOGE
    global EVAL_ROWS, EVAL_ANCHORS, EVAL_SOURCE_PATHS, EVAL_ORIENTED, EVAL_HAS_TARGET, EVAL_BALL_RADIUS
    if not os.path.isfile(report_path):
        raise FileNotFoundError(report_path)
    if not os.path.isdir(dataset_dir):
        raise NotADirectoryError(dataset_dir)
    report = json.load(open(report_path, encoding="utf-8"))
    rows = report.get("predictions") if isinstance(report, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{report_path} has no prediction rows")
    required = {"file", "pred_normal"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"{report_path} has incomplete prediction rows")
    EVAL_HAS_TARGET = all("target_normal" in row and "axis_error_deg" in row for row in rows)
    EVAL_ROWS = {row["file"]: row for row in rows}
    EVAL_ORIENTED = report.get("metadata", {}).get("normal_convention") == "oriented_toward_camera"
    EVAL_BALL_RADIUS = report.get("metadata", {}).get("ball_radius_m")
    if EVAL_BALL_RADIUS is not None:
        EVAL_BALL_RADIUS = float(EVAL_BALL_RADIUS)
    anchors_path = next(
        (os.path.join(dataset_dir, name) for name in ("anchors_manual3d.json", "anchors_obb.json")
         if os.path.isfile(os.path.join(dataset_dir, name))),
        None,
    )
    EVAL_ANCHORS = json.load(open(anchors_path, encoding="utf-8")) if anchors_path else {}
    source_root = source_root or os.path.join(os.path.dirname(dataset_dir), "fuelcap_pass_20260803_10847")
    manifest_path = os.path.join(dataset_dir, "manifest.jsonl")
    manifest = [json.loads(line) for line in open(manifest_path, encoding="utf-8") if line.strip()]
    EVAL_SOURCE_PATHS = {
        item["file"]: os.path.join(source_root, item["source_cloud"])
        for item in manifest if item["file"] in EVAL_ROWS
    }
    PCD = os.path.join(dataset_dir, "clouds")
    FILES = [row["file"] for row in rows]
    NORMS = [row.get("target_normal", row["pred_normal"]) for row in rows]
    CONF = [report_consensus(row) for row in rows]
    AGREE = [report_error(row) if EVAL_HAS_TARGET else 0.0 for row in rows]
    NINNER = [0] * len(rows)
    TIER = {row["file"]: "test" for row in rows}
    PROD = {row["file"]: {"normal": row["pred_normal"]} for row in rows}
    MOGE = {}
    focus_base = (arg_value("--focus-file") or "").strip()
    if focus_base:
        if not focus_base.endswith(".npz"):
            focus_base += ".npz"
        if focus_base not in EVAL_ROWS:
            raise ValueError(f"--focus-file is absent from report: {focus_base}")
        FILES = [focus_base]
        NORMS = [EVAL_ROWS[focus_base].get("target_normal", EVAL_ROWS[focus_base]["pred_normal"])]
        CONF = [report_consensus(EVAL_ROWS[focus_base])]
        AGREE = [report_error(EVAL_ROWS[focus_base]) if EVAL_HAS_TARGET else 0.0]
        NINNER = [0]
    print(f"MSECNet evaluation viewer: {len(FILES)} samples from {os.path.basename(report_path)}", flush=True)
    return report.get("summary", {})


if EVAL_REPORT_PATH:
    if not EVAL_DATASET_DIR:
        raise ValueError("--msecnet-dataset is required with --msecnet-report")
    configure_evaluation(
        EVAL_REPORT_PATH,
        EVAL_DATASET_DIR,
        arg_value("--msecnet-source-root") or None,
    )

# index source images by basename
SRC = {}
for _p in glob.glob(os.path.join(SRC_DIR, "**", "*.png"), recursive=True):
    SRC[os.path.splitext(os.path.basename(_p))[0]] = _p


def src_path(f):
    stem = f[:-4] if f.endswith(".npz") else f
    return SRC.get(stem) or SRC.get(stem.split("__", 1)[-1])


sys.path.insert(0, os.path.join(ROOT, "shared"))
import cap_patch  # noqa: E402
from make_normal_labels import ransac_plane  # noqa: E402

_KC = {"mtime": 0, "data": {}}


def knob_centers():
    """knob centers, reloaded if the json changed."""
    if os.path.exists(KCJSON):
        mt = os.path.getmtime(KCJSON)
        if mt != _KC["mtime"]:
            try:
                _KC["data"] = json.load(open(KCJSON)); _KC["mtime"] = mt
            except Exception:
                pass
    return _KC["data"]


def prefetch(i, n=3):
    for j in range(i + 1, min(i + 1 + n, len(FILES))):
        f = FILES[j]; src = src_path(f)
        if not src:
            continue
        out = os.path.join(CACHE_DIR, f + ".npz")
        if os.path.exists(out) or os.path.exists(os.path.join(QDIR, f + ".pre")) \
                or os.path.exists(os.path.join(QDIR, f + ".req")):
            continue
        json.dump({"img": src, "txt": os.path.splitext(src)[0] + ".txt", "out": out},
                  open(os.path.join(QDIR, f + ".pre"), "w"))


def _flipYZ(v):   # CV camera frame -> three.js GL frame; self-inverse
    return [float(v[0]), -float(v[1]), -float(v[2])]


# Real-time FULL-image dense cloud (cached). Requires moge_worker.py running.
# Set FULL_CLOUD=0 to skip and always use the ROI crop (fast, lower quality).
CACHE_DIR = "/tmp/fuelcap_fcc"
QDIR = os.path.join(CACHE_DIR, "_queue")
os.makedirs(QDIR, exist_ok=True)
_FC_ENABLED = os.environ.get("FULL_CLOUD", "1") != "0"
# Moge takes 2-4s per image. If worker isn't running, don't block the UI — fall back immediately.
_FC_TIMEOUT = float(os.environ.get("FULL_CLOUD_TIMEOUT", "1"))


def full_cloud_path(f, timeout=_FC_TIMEOUT):
    if not _FC_ENABLED:
        return None
    src = src_path(f)
    if not src:
        return None
    out = os.path.join(CACHE_DIR, f + ".npz")
    if os.path.exists(out):
        return out
    txt = os.path.splitext(src)[0] + ".txt"
    req = os.path.join(QDIR, f + ".req")
    if not os.path.exists(req):
        json.dump({"img": src, "txt": txt, "out": out}, open(req, "w"))
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(out):
            return out
        if os.path.exists(out + ".err"):
            return None
        time.sleep(0.2)
    return None


_NPZ_LRU = {}
_NPZ_ORDER = []
_PLY_LRU = {}
_PLY_ORDER = []
_META_LRU = {}
_META_ORDER = []
PLY_DTYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
}


def load_npz(path):
    d = _NPZ_LRU.get(path)
    if d is None:
        z = np.load(path)
        d = {k: z[k] for k in z.files}
        _NPZ_LRU[path] = d; _NPZ_ORDER.append(path)
        if len(_NPZ_ORDER) > 12:
            _NPZ_LRU.pop(_NPZ_ORDER.pop(0), None)
    return d


def load_ply(path):
    """Load XYZ and optional RGB from a standard binary little-endian PLY."""
    cached = _PLY_LRU.get(path)
    if cached is not None:
        return cached
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        vertex_count, properties, in_vertex = None, [], False
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            fields = line.split()
            if fields[:2] == ["element", "vertex"]:
                vertex_count, in_vertex = int(fields[2]), True
            elif fields[:1] == ["element"]:
                in_vertex = False
            elif in_vertex and fields[:1] == ["property"]:
                if len(fields) != 3 or fields[1] not in PLY_DTYPES:
                    raise ValueError(f"unsupported PLY vertex property: {line}")
                properties.append((fields[2], "<" + PLY_DTYPES[fields[1]]))
        if vertex_count is None or not {"x", "y", "z"}.issubset(dict(properties)):
            raise ValueError(f"PLY has no XYZ vertex fields: {path}")
        vertices = np.fromfile(f, dtype=np.dtype(properties), count=vertex_count)
    data = {"xyz": np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)}
    if {"red", "green", "blue"}.issubset(vertices.dtype.names):
        data["rgb"] = np.column_stack((vertices["red"], vertices["green"], vertices["blue"])).astype(np.float32) / 255.0
    _PLY_LRU[path] = data
    _PLY_ORDER.append(path)
    if len(_PLY_ORDER) > 2:
        _PLY_LRU.pop(_PLY_ORDER.pop(0), None)
    return data


def load_source_cloud(path):
    return load_ply(path) if path.lower().endswith(".ply") else load_npz(path)


app = Flask(__name__, static_folder=None)


def labels():
    return json.load(open(SAVE)) if os.path.exists(SAVE) else {}


def flagged():
    return json.load(open(FLAG)) if os.path.exists(FLAG) else {}


def available_evaluation_options():
    """Return repository-owned checkpoints and compatible prepared datasets."""
    project_root = Path(ROOT)
    checkpoints = []
    for path in sorted((project_root / "msecnet_best").glob("out/*/best.pt")):
        checkpoints.append({
            "id": str(path.relative_to(project_root)),
            "label": f"pseudo-OBB: {path.parent.name}",
            "model_type": "pseudo_obb",
        })
    bundled = project_root / "msecnet_best" / "checkpoints" / "best.pt"
    if bundled.is_file():
        checkpoints.append({
            "id": str(bundled.relative_to(project_root)),
            "label": "pseudo-OBB: bundled historical best",
            "model_type": "pseudo_obb",
        })
    for path in sorted((project_root / "msecnet_ball" / "out").glob("*/best.pt")):
        checkpoints.append({
            "id": str(path.relative_to(project_root)),
            "label": f"8 cm ball: {path.parent.name}",
            "model_type": "center_ball",
        })

    datasets = []
    source_candidates = list((project_root / "data").glob("fuelcap_pass_*"))
    source_candidates.extend((project_root / "raw_data").glob("*/open"))
    for path in sorted((project_root / "data").iterdir()):
        labeled_required = ("labels_manual3d.npz", "clouds", "anchors_manual3d.json", "manifest.jsonl")
        unlabeled_required = ("unlabeled_test.npz", "clouds", "anchors_obb.json", "split.json", "manifest.jsonl")
        is_labeled = path.is_dir() and all((path / name).exists() for name in labeled_required)
        is_unlabeled = path.is_dir() and all((path / name).exists() for name in unlabeled_required)
        split_by_car_model = path / "split_by_car_model.json"
        split_by_group = path / "split_by_generalization_group.json"
        if is_labeled and split_by_car_model.is_file():
            model_type, model_types, split_path = "pseudo_obb", ("pseudo_obb",), split_by_car_model
        elif is_labeled and split_by_group.is_file():
            anchors = json.loads((path / "anchors_manual3d.json").read_text(encoding="utf-8"))
            selection = next(iter(anchors.values()), {}).get("selection")
            if selection != "manual_center_ball_patch":
                continue
            model_type, model_types, split_path = "center_ball", ("center_ball",), split_by_group
        elif is_unlabeled:
            # A detector-derived OBB crop has no reviewed 3D center. The ball
            # predictor uses its crop centroid as the requested proxy center.
            model_type, model_types, split_path = "pseudo_obb", ("pseudo_obb", "center_ball"), path / "split.json"
        else:
            continue
        manifest = [json.loads(line) for line in (path / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if not manifest:
            continue
        source_cloud = manifest[0].get("source_cloud", "")
        source_root = next(
            (candidate for candidate in sorted(source_candidates)
             if (candidate / source_cloud).is_file()),
            None,
        )
        split = json.loads(split_path.read_text(encoding="utf-8"))
        splits = [name for name in ("train", "val", "test") if split.get(name)]
        if splits:
            datasets.append({
                "id": path.name,
                "label": f"unlabeled OBB: {path.name}" if is_unlabeled else path.name,
                "splits": splits,
                "source_root": str(source_root) if source_root else "",
                "kind": "labeled" if is_labeled else "unlabeled",
                "model_type": model_type,
                "model_types": list(model_types),
            })
    supported_model_types = {item["model_type"] for item in checkpoints}
    datasets = [item for item in datasets if item["model_type"] in supported_model_types]
    return {"checkpoints": checkpoints, "datasets": datasets}


def current_job():
    with EVAL_STATE_LOCK:
        job = dict(INFERENCE_JOB)
        job["logs"] = list(INFERENCE_JOB.get("logs", [])[-80:])
        return job


def append_job_log(message):
    with EVAL_STATE_LOCK:
        logs = INFERENCE_JOB.setdefault("logs", [])
        logs.append(str(message).rstrip()[:1200])
        del logs[:-200]


def run_inference_job(command, report_path, dataset_dir, source_root):
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with EVAL_STATE_LOCK:
            INFERENCE_JOB["pid"] = process.pid
        for line in process.stdout:
            append_job_log(line)
        if process.wait() != 0:
            raise RuntimeError(f"infer.py exited with status {process.returncode}")
        with EVAL_STATE_LOCK:
            summary = configure_evaluation(str(report_path), str(dataset_dir), str(source_root))
            INFERENCE_JOB.update({"state": "complete", "summary": summary, "pid": None})
        append_job_log("Inference completed. The viewer now shows this result.")
    except Exception as exc:
        append_job_log(f"ERROR: {exc}")
        with EVAL_STATE_LOCK:
            INFERENCE_JOB.update({"state": "failed", "error": str(exc), "pid": None})


@app.route("/api/msecnet/options")
def msecnet_options():
    if not MSECNET_UI_MODE:
        abort(404)
    return jsonify(available_evaluation_options())


@app.route("/api/msecnet/job")
def msecnet_job():
    if not MSECNET_UI_MODE:
        abort(404)
    return jsonify(current_job())


@app.route("/api/msecnet/job", methods=["POST"])
def start_msecnet_job():
    if not MSECNET_UI_MODE:
        abort(404)
    request_data = request.get_json(silent=True) or {}
    options = available_evaluation_options()
    checkpoint_by_id = {item["id"]: item for item in options["checkpoints"]}
    dataset_by_id = {item["id"]: item for item in options["datasets"]}
    checkpoint_id = request_data.get("checkpoint")
    dataset_id = request_data.get("dataset")
    split_name = request_data.get("split")
    if checkpoint_id not in checkpoint_by_id or dataset_id not in dataset_by_id:
        return jsonify({"error": "unknown checkpoint or dataset"}), 400
    dataset = dataset_by_id[dataset_id]
    if split_name not in dataset["splits"]:
        return jsonify({"error": "unknown or empty split"}), 400
    checkpoint_type = checkpoint_by_id[checkpoint_id]["model_type"]
    if checkpoint_type not in dataset.get("model_types", [dataset["model_type"]]):
        return jsonify({"error": "checkpoint and dataset use different input definitions"}), 400

    checkpoint = Path(ROOT) / checkpoint_id
    dataset_dir = Path(ROOT) / "data" / dataset_id
    source_root = Path(dataset["source_root"]) if dataset["source_root"] else dataset_dir
    output_dir = checkpoint.parent / "web_inference" / dataset_id / split_name
    if dataset["kind"] == "unlabeled" and checkpoint_type == "center_ball":
        if not dataset["source_root"]:
            return jsonify({"error": "ball inference needs the original source-cloud directory"}), 400
        command = [
            sys.executable, "-u", str(Path(ROOT) / "msecnet_ball" / "predict_unlabeled.py"),
            str(checkpoint), str(dataset_dir), "--source-root", str(source_root), "--out", str(output_dir),
        ]
    elif dataset["kind"] == "unlabeled":
        command = [
            sys.executable, "-u", str(Path(ROOT) / "msecnet_best" / "predict_unlabeled.py"),
            str(checkpoint), str(dataset_dir), "--out", str(output_dir),
        ]
    elif checkpoint_type == "center_ball":
        command = [
            sys.executable, "-u", str(Path(ROOT) / "msecnet_ball" / "infer.py"),
            str(checkpoint), str(dataset_dir / "labels_manual3d.npz"), str(dataset_dir / "clouds"),
            "--centers", str(dataset_dir / "anchors_manual3d.json"),
            "--split", str(dataset_dir / "split_by_generalization_group.json"),
            "--split-name", split_name, "--out", str(output_dir),
        ]
    elif dataset["kind"] == "labeled":
        command = [
            sys.executable, "-u", str(Path(ROOT) / "msecnet_best" / "infer.py"),
            str(checkpoint), str(dataset_dir / "labels_manual3d.npz"), str(dataset_dir / "clouds"),
            "--centers", str(dataset_dir / "anchors_manual3d.json"),
            "--split", str(dataset_dir / "split_by_car_model.json"),
            "--split-name", split_name, "--out", str(output_dir),
        ]
    with EVAL_STATE_LOCK:
        if INFERENCE_JOB.get("state") == "running":
            return jsonify({"error": "an inference job is already running"}), 409
        INFERENCE_JOB.clear()
        INFERENCE_JOB.update({
            "state": "running", "checkpoint": checkpoint_id, "dataset": dataset_id,
            "split": split_name, "output_dir": str(output_dir), "logs": ["Starting infer.py..."],
        })
    threading.Thread(
        target=run_inference_job,
        args=(command, output_dir / "report.json", dataset_dir, source_root),
        daemon=True,
    ).start()
    return jsonify(current_job()), 202


def make_photo(xyz, rgb, inner, K, W, H, S=512):
    Z = xyz[:, 2].copy(); Z[Z < 1e-3] = 1e-3
    fx, fy, cx, cy = K[0, 0] * W, K[1, 1] * H, K[0, 2] * W, K[1, 2] * H
    u = fx * xyz[:, 0] / Z + cx; v = fy * xyz[:, 1] / Z + cy
    umin, umax = np.percentile(u, [0.5, 99.5]); vmin, vmax = np.percentile(v, [0.5, 99.5])
    span = max(umax - umin, vmax - vmin, 1e-6) * 1.12
    ui = np.clip(((u - (umin + umax) / 2) / span + 0.5) * S, 0, S - 1).astype(int)
    vi = np.clip(((v - (vmin + vmax) / 2) / span + 0.5) * S, 0, S - 1).astype(int)
    lo = np.percentile(rgb, 2, 0); hi = np.percentile(rgb, 98, 0)
    rs = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1) * 0.9 + 0.08
    photo = np.zeros((S, S, 3), np.uint8); o = np.argsort(-Z)
    for dd in ((0, 0), (0, 1), (1, 0), (1, 1)):
        photo[np.clip(vi[o] + dd[0], 0, S - 1), np.clip(ui[o] + dd[1], 0, S - 1)] = (rs[o] * 255).astype(np.uint8)
    if inner.sum() > 10:
        iu, iv = ui[inner], vi[inner]
        cv2.rectangle(photo, (int(iu.min()), int(iv.min())), (int(iu.max()), int(iv.max())), (0, 255, 0), 2)
        cv2.putText(photo, "inner_cover", (int(iu.min()), max(14, int(iv.min()) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return photo


def evaluation_cloud(i):
    """Serve explicit OBB/ball inference inputs or a larger source-cloud context."""
    f = FILES[i]
    row = EVAL_ROWS[f]
    training_cloud = load_npz(os.path.join(PCD, f))
    training_xyz = training_cloud["xyz"].astype(np.float32)
    anchor = EVAL_ANCHORS.get(f, {})
    center = np.asarray(
        row.get("center_3d", anchor.get("center_3d", np.median(training_xyz, axis=0))), dtype=np.float32
    )
    mode = request.args.get("mode", "inference_obb")
    input_geometry = "center_ball" if EVAL_BALL_RADIUS is not None else "pseudo_obb"
    ball_radius_m = EVAL_BALL_RADIUS or DISPLAY_BALL_RADIUS_M
    # Keep the previous URL working while making the two model input views explicit.
    if mode == "patch":
        mode = "inference_ball" if input_geometry == "center_ball" else "inference_obb"
    source_path = EVAL_SOURCE_PATHS.get(f)
    if mode in ("whole", "sphere") and source_path and os.path.isfile(source_path):
        source_cloud = load_source_cloud(source_path)
        source_xyz = source_cloud["xyz"].astype(np.float32)
        keep = np.ones(len(source_xyz), dtype=bool)
        if mode == "sphere":
            keep = np.linalg.norm(source_xyz - center, axis=1) < 0.5
            if keep.sum() >= 80:
                source_xyz = source_xyz[keep]
            else:
                keep = np.ones(len(source_xyz), dtype=bool)
        xyz = source_xyz
        source_rgb = source_cloud.get("rgb")
        if source_rgb is None:
            rgb = np.tile(np.asarray((0.62, 0.72, 0.88), dtype=np.float32), (len(xyz), 1))
        else:
            rgb = source_rgb[keep].astype(np.float32)
            if rgb.max() > 1.0:
                rgb /= 255.0
        max_display_points = 120000 if mode == "whole" else 160000
        if len(xyz) > max_display_points:
            indices = np.random.default_rng(0).choice(len(xyz), max_display_points, replace=False)
            xyz, rgb = xyz[indices], rgb[indices]
        mode_name = "source_0.5m" if mode == "sphere" else "source_full"
    elif mode == "inference_ball":
        # The OBB crop supplies the proxy center for unlabeled data, but the
        # actual ball model input comes from the uncropped source point cloud.
        if source_path and os.path.isfile(source_path):
            source_cloud = load_source_cloud(source_path)
            source_xyz = source_cloud["xyz"].astype(np.float32)
            ball_mask = np.linalg.norm(source_xyz - center, axis=1) <= ball_radius_m
            xyz = source_xyz[ball_mask]
            source_rgb = source_cloud.get("rgb")
            if source_rgb is None:
                rgb = np.tile(np.asarray((0.62, 0.72, 0.88), dtype=np.float32), (len(xyz), 1))
            else:
                rgb = source_rgb[ball_mask].astype(np.float32)
                if rgb.max() > 1.0:
                    rgb /= 255.0
        else:
            xyz = np.empty((0, 3), dtype=np.float32)
        if not len(xyz):
            # Do not silently present an OBB crop as a ball-model input.
            xyz = training_xyz
            rgb = np.tile(np.asarray((0.62, 0.72, 0.88), dtype=np.float32), (len(xyz), 1))
            mode_name = "center_ball_unavailable"
        else:
            mode_name = "center_ball_inference"
    else:
        # The prepared cloud is the pseudo-OBB model input. It is also useful as
        # a comparison view beside the ball subset for a center-ball prediction.
        xyz = training_xyz
        rgb = np.tile(np.asarray((0.62, 0.72, 0.88), dtype=np.float32), (len(xyz), 1))
        mode_name = "obb_inference"
    scale = float(np.percentile(np.linalg.norm(xyz - center, axis=1), 97) + 1e-9)
    pts = ((xyz - center) / scale) * np.array([1, -1, -1], dtype=np.float32)
    has_target = "target_normal" in row
    target = np.asarray(row["target_normal"] if has_target else row["pred_normal"], dtype=np.float32)
    prediction = np.asarray(row["pred_normal"], dtype=np.float32)
    target /= np.linalg.norm(target) + 1e-9
    prediction /= np.linalg.norm(prediction) + 1e-9
    # Legacy reports were sign-invariant; new reports preserve this visible orientation.
    if has_target and not EVAL_ORIENTED and float(prediction @ target) < 0:
        prediction = -prediction
    rect = None
    if "rectangle_wh_m" in anchor:
        width, height = np.asarray(anchor["rectangle_wh_m"], dtype=np.float32) / scale
        tangent = np.asarray(anchor.get("tangent", (1, 0, 0)), dtype=np.float32)
        rect = {
            "w": float(width), "h": float(height),
            "dir": _flipYZ(tangent / (np.linalg.norm(tangent) + 1e-9)),
        }
    header = {
        "i": i, "n": len(FILES), "file": f, "conf": CONF[i], "agree": AGREE[i], "ninner": len(training_xyz),
        "labeled_count": 0, "tier": "test", "has_src": False, "mode": mode_name,
        "npts": int(len(pts)), "fullcloud": mode == "whole", "flagged": None, "knob_rect": rect,
        "input_geometry": input_geometry,
        "is_actual_input": (mode == "inference_obb" and input_geometry == "pseudo_obb")
        or (mode == "inference_ball" and input_geometry == "center_ball"),
        "ball_radius_m": ball_radius_m,
        "quality": report_consensus(row), "v3_agree": report_error(row) if has_target else None,
        "normal": _flipYZ(target), "prelabel": _flipYZ(target),
        "model_label": _flipYZ(prediction), "moge_label": None, "is_saved": False, "photo": "",
        "readonly": True, "car_model": row.get("car_model", ""),
        "has_target": has_target, "prediction_error_deg": report_error(row) if has_target else None,
    }
    inter = np.empty((len(pts), 6), np.float32)
    inter[:, :3] = pts; inter[:, 3:] = rgb
    hb = json.dumps(header).encode()
    hb += b" " * ((-(4 + len(hb))) % 4)
    return Response(struct.pack("<I", len(hb)) + hb + inter.tobytes(), mimetype="application/octet-stream")


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/vendor/<path:p>")
def vendor(p):
    return send_from_directory(os.path.join(HERE, "vendor"), p)


@app.route("/api/meta")
def meta():
    input_geometry = "center_ball" if EVAL_BALL_RADIUS is not None else "pseudo_obb"
    return jsonify({
        "n": len(FILES), "labeled": list(labels().keys()), "evaluation": EVAL_MODE,
        "input_geometry": input_geometry if EVAL_MODE and FILES else None,
    })


@app.route("/api/cloud/<int:i>")
def cloud(i):
    if EVAL_MODE:
        with EVAL_STATE_LOCK:
            if i < 0 or i >= len(FILES):
                abort(404, "no completed inference result is loaded")
            return evaluation_cloud(i)
    if i < 0 or i >= len(FILES):
        abort(404)
    f = FILES[i]
    fc = full_cloud_path(f)
    d = load_npz(fc if fc else os.path.join(PCD, f))
    fullcloud = fc is not None
    lab = d["label"]; xa = d["xyz"].astype(np.float32); ra = d["rgb"].astype(np.float32) / 255.0
    inner = lab == 1
    W, H = int(d["w"]), int(d["h"])
    cap_xyz = xa[inner]; cap_rgb = ra[inner]
    mode = request.args.get("mode", "patch")
    # The explicit inference views apply to read-only MSECNet evaluation. Keep
    # the original labeler behavior unchanged when this page is used for edits.
    if mode in ("inference_obb", "inference_ball"):
        mode = "patch"
    kc = knob_centers().get(f)
    # --- per-cloud meta ---
    meta = _META_LRU.get(f)
    if meta is None:
        anchor = cap_patch.knob_anchor(cap_xyz, d["K_norm"], W, H, kc) if (kc is not None and len(cap_xyz) >= 50) else None
        prelabel = list(NORMS[i])
        if anchor is not None:
            pm0 = cap_patch.cap_patch_mask(cap_xyz, anchor, radius_frac=0.3)
            pset = cap_xyz[pm0] if pm0.sum() >= 60 else cap_xyz
            sp = np.linalg.norm(np.percentile(pset, 97, 0) - np.percentile(pset, 3, 0)) + 1e-9
            pn, _, _ = ransac_plane(pset, sp * 0.02)
            if pn @ (-anchor) < 0:
                pn = -pn
            prelabel = [float(x) for x in pn]
        photo = make_photo(xa, ra, inner, d["K_norm"], W, H)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        meta = {"anchor": anchor, "prelabel": prelabel,
                "photo": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}
        _META_LRU[f] = meta; _META_ORDER.append(f)
        if len(_META_ORDER) > 12:
            _META_LRU.pop(_META_ORDER.pop(0), None)
    anchor = meta["anchor"]; prelabel = meta["prelabel"]; photo_b64 = meta["photo"]
    c = anchor if anchor is not None else np.median(cap_xyz, 0)
    # displayed points by mode
    if mode == "sphere" and anchor is not None:
        sel = np.linalg.norm(xa - anchor, axis=1) < 0.5
        xyz, rgb, modename = xa[sel], ra[sel], "sphere"
        scale = float(np.percentile(np.linalg.norm(xyz - c, axis=1), 97) + 1e-9)
    elif mode == "patch" and anchor is not None and cap_patch.cap_patch_mask(cap_xyz, anchor, 0.5).sum() >= 60:
        pm = cap_patch.cap_patch_mask(cap_xyz, anchor, radius_frac=0.5)
        xyz, rgb, modename = cap_xyz[pm], cap_rgb[pm], "patch"
        scale = float(np.percentile(np.linalg.norm(cap_xyz - c, axis=1), 97) + 1e-9)
    else:
        xyz, rgb, modename = cap_xyz, cap_rgb, "whole"
        scale = float(np.percentile(np.linalg.norm(cap_xyz - c, axis=1), 97) + 1e-9)
    if len(xyz) > 80000:
        k = np.random.RandomState(0).choice(len(xyz), 80000, replace=False); xyz, rgb = xyz[k], rgb[k]
    pts = ((xyz - c) / scale) * np.array([1, -1, -1], np.float32)  # CV->GL
    lo = np.percentile(rgb, 3, 0); hi = np.percentile(rgb, 97, 0)
    rs = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    labs = labels()
    saved = labs.get(f)
    if fullcloud:
        prefetch(i)
    knob_rect = None
    if anchor is not None and kc is not None:
        K = d["K_norm"]; az = max(float(anchor[2]), 1e-3)
        fxp, fyp, cxp, cyp = K[0, 0] * W, K[1, 1] * H, K[0, 2] * W, K[1, 2] * H
        a_ang = float(kc.get("angle", 0.0)); kxp, kyp = kc["cxf"] * W, kc["cyf"] * H

        def _bp(px, py):
            return np.array([(px - cxp) * az / fxp, (py - cyp) * az / fyp, az])
        dir3 = _bp(kxp + 30 * np.cos(a_ang), kyp + 30 * np.sin(a_ang)) - _bp(kxp, kyp)
        nrm = np.array(prelabel, float); nrm /= np.linalg.norm(nrm) + 1e-9
        dpl = dir3 - (dir3 @ nrm) * nrm
        dpl /= np.linalg.norm(dpl) + 1e-9
        knob_rect = {"w": float(kc.get("wf", 0.05) * az / (K[0, 0] * scale)),
                     "h": float(kc.get("hf", 0.05) * az / (K[1, 1] * scale)),
                     "dir": [float(dpl[0]), -float(dpl[1]), -float(dpl[2])]}
    # BINARY protocol
    header = {
        "i": i, "n": len(FILES), "file": f, "conf": CONF[i], "agree": AGREE[i], "ninner": NINNER[i],
        "labeled_count": len(labs),
        "tier": TIER.get(f),
        "has_src": src_path(f) is not None, "mode": modename, "npts": int(len(pts)), "fullcloud": fullcloud,
        "flagged": flagged().get(f), "knob_rect": knob_rect,
        "quality": V3.get(f, {}).get("inlier"), "v3_agree": V3.get(f, {}).get("agree"),
        "normal": _flipYZ(saved if saved else prelabel), "prelabel": _flipYZ(prelabel),
        "model_label": (_flipYZ(PROD[f]["normal"]) if f in PROD else None),
        "moge_label": (_flipYZ(MOGE[f]["normal"]) if f in MOGE else None),
        "is_saved": saved is not None,
        "photo": photo_b64,
    }
    inter = np.empty((len(pts), 6), np.float32)
    inter[:, :3] = pts.astype(np.float32); inter[:, 3:] = rs.astype(np.float32)
    hb = json.dumps(header).encode()
    hb += b" " * ((-(4 + len(hb))) % 4)
    body = struct.pack("<I", len(hb)) + hb + inter.tobytes()
    return Response(body, mimetype="application/octet-stream")


@app.route("/api/srcimg/<int:i>")
def srcimg(i):
    f = FILES[i]; p = src_path(f)
    if not p or not os.path.exists(p):
        return ("", 404)
    im = cv2.imread(p)
    if im is None:
        return ("", 404)
    H, W = im.shape[:2]
    d = np.load(os.path.join(PCD, f)); inner = d["label"] == 1
    if inner.sum() > 10:
        xyz = d["xyz"][inner]; K = d["K_norm"]; Z = xyz[:, 2].copy(); Z[Z < 1e-3] = 1e-3
        u = K[0, 0] * W * xyz[:, 0] / Z + K[0, 2] * W
        v = K[1, 1] * H * xyz[:, 1] / Z + K[1, 2] * H
        cv2.rectangle(im, (int(u.min()), int(v.min())), (int(u.max()), int(v.max())),
                      (0, 255, 0), max(2, W // 300))
    sc = 760.0 / max(W, H)
    im = cv2.resize(im, (int(W * sc), int(H * sc)))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/api/flag", methods=["POST"])
def flag():
    j = request.get_json()
    fl = flagged(); fl[j["file"]] = j.get("flag", "abnormal")
    json.dump(fl, open(FLAG, "w"))
    return jsonify({"ok": True, "count": len(fl)})


@app.route("/api/save", methods=["POST"])
def save():
    if EVAL_MODE:
        return jsonify({"ok": False, "error": "MSECNet evaluation viewer is read-only"}), 403
    j = request.get_json()
    labs = labels(); labs[j["file"]] = _flipYZ(j["normal"])
    json.dump(labs, open(SAVE, "w"))
    return jsonify({"ok": True, "count": len(labs)})


if __name__ == "__main__":
    port = int(arg_value("--port", "8765"))
    print(f"serving {len(FILES)} clouds at http://0.0.0.0:{port}  (labels -> {SAVE})", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True)
