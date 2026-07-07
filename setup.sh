#!/bin/bash
# === point2normal — complete conda environment setup ===
# Usage: bash setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="point2normal"

echo "========================================"
echo " point2normal — conda environment setup"
echo " target: $ROOT"
echo " env:    $ENV_NAME"
echo "========================================"

# ---- 1. create conda environment ----
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/5] conda env '${ENV_NAME}' already exists, skipping create"
else
    echo "[1/5] creating conda environment from environment.yml ..."
    conda env create -f "$ROOT/environment.yml"
fi

# get the python path
CONDA_PY=$(conda run -n "$ENV_NAME" which python)
echo "       python: $CONDA_PY"

# ---- 2. verify key packages ----
echo "[2/5] verifying key packages ..."
conda run -n "$ENV_NAME" python -c "
import torch; print(f'  PyTorch {torch.__version__}  CUDA {torch.version.cuda}  GPU available: {torch.cuda.is_available()}')
import numpy; print(f'  numpy {numpy.__version__}')
import cv2; print(f'  opencv {cv2.__version__}')
import yaml; print(f'  pyyaml available')
import trimesh; print(f'  trimesh {trimesh.__version__}')
import ultralytics; print(f'  ultralytics {ultralytics.__version__}')
import moge; print(f'  moge available')
"

# ---- 3. compile pointops_cuda (MSECNet dependency) ----
echo "[3/5] compiling pointops_cuda ..."
POINTOPS_DIR="$ROOT/msecnet/MSECNet/scripts/lib/pointops"
if [ -f "$POINTOPS_DIR/build/lib.linux-x86_64-cpython-311/pointops_cuda"*.so ]; then
    echo "       pointops_cuda already compiled, skipping"
else
    # The system has CUDA 12.5; PyTorch was built with CUDA 12.4.
    # torch.utils.cpp_extension will use the CUDA that PyTorch was built with.
    cd "$POINTOPS_DIR"
    CC=/usr/bin/gcc CXX=/usr/bin/g++ conda run -n "$ENV_NAME" python setup.py install
    echo "       pointops_cuda compiled successfully"
fi

# ---- 4. create data symlinks ----
echo "[4/5] setting up data symlinks ..."
DATA_DIR="$ROOT/data"
mkdir -p "$DATA_DIR"

# Symlink the pcd_dataset_roi if it exists
PCD_SRC="/data2/shendu/code/ruoyu/fuelcap_6dpose/pipeline/pcd_dataset_roi"
if [ -d "$PCD_SRC" ] && [ ! -e "$DATA_DIR/pcd_dataset_roi" ]; then
    ln -s "$PCD_SRC" "$DATA_DIR/pcd_dataset_roi"
    echo "       linked $DATA_DIR/pcd_dataset_roi -> $PCD_SRC"
elif [ -e "$DATA_DIR/pcd_dataset_roi" ]; then
    echo "       pcd_dataset_roi already exists, skipping"
else
    echo "       [WARN] pcd_dataset_roi not found at $PCD_SRC"
    echo "       please manually symlink or copy your dataset to $DATA_DIR/pcd_dataset_roi"
fi

# Symlink models directory
MODELS_SRC="/data2/shendu/code/ruoyu/fuelcap_6dpose/models"
if [ -d "$MODELS_SRC" ] && [ ! -e "$ROOT/models" ]; then
    ln -s "$MODELS_SRC" "$ROOT/models"
    echo "       linked $ROOT/models -> $MODELS_SRC"
elif [ -e "$ROOT/models" ]; then
    echo "       models already exists, skipping"
else
    echo "       [WARN] models not found at $MODELS_SRC"
    echo "       please manually symlink or copy your models (YOLO .pt files, hf_cache)"
fi

# ---- 5. create output directories ----
echo "[5/5] creating output directories ..."
mkdir -p "$ROOT/output/normal_pred_demo"
mkdir -p "$ROOT/output/runs_infer"

echo ""
echo "========================================"
echo " setup complete!"
echo ""
echo " activate:  conda activate $ENV_NAME"
echo ""
echo " quick start:"
echo "   1. generate labels:"
echo "      python shared/make_normal_labels.py data/pcd_dataset_roi --radius 0.3 --out shared/normal_labels_patch03.npz"
echo ""
echo "   2. train NormalNet:"
echo "      python normalnet/train.py shared/normal_labels_patch03.npz data/pcd_dataset_roi --soft --steps 80000 --out normalnet/ckpt_normal"
echo ""
echo "   3. train MSECNet:"
echo "      python msecnet/train_v1.py shared/normal_labels_patch03.npz data/pcd_dataset_roi --soft --steps 20000 --out msecnet/ckpt_msecnet"
echo ""
echo "   4. run inference:"
echo "      python normalnet/infer_normal.py path/to/image.png"
echo "========================================"
