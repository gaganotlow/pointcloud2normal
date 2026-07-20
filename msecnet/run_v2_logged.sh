#!/usr/bin/env bash
# Logged MSECNet geometry-only training.
#
# This is the pure-geometry counterpart of msecnet_add_rgb/train_rgb_fusion.py:
# it writes train.log, metrics.csv/jsonl, curves.png, run_config.json,
# val_predictions/*.json, and val_vis/*.png.
#
# Usage:
#   cd /data2/shendu/code/ruoyu/train_point2normal
#   bash msecnet/run_v2_logged.sh

set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

LABELS="${LABELS:-shared/normal_labels_patch03.npz}"
PCD_DIR="${PCD_DIR:-data/pcd_dataset_roi}"
OUT_DIR="${OUT_DIR:-msecnet/ckpt_msecnet_v2_logged}"
GPU="${GPU:-0}"

STEPS="${STEPS:-12000}"
TAIL_STEPS="${TAIL_STEPS:-0}"
BS="${BS:-12}"
MAX_POINTS="${MAX_POINTS:-0}"
AUG_DEG="${AUG_DEG:-0}"
LR="${LR:-5e-4}"
WD="${WD:-1e-4}"
VAL_EVERY="${VAL_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-100}"
VIS_EVERY="${VIS_EVERY:-1000}"
VIS_SAMPLES="${VIS_SAMPLES:-6}"

check_path() {
    local path="$1"
    local desc="$2"
    if [[ ! -e "$path" ]]; then
        echo "ERROR: missing $desc: $path" >&2
        exit 1
    fi
}

check_path "$LABELS" "labels npz"
check_path "$PCD_DIR" "point-cloud directory"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "MSECNet v2 geometry-only logged training"
echo "=========================================="
echo "GPU:             $GPU"
echo "Output:          $OUT_DIR"
echo "Steps/tail:      $STEPS / $TAIL_STEPS"
echo "Batch/maxpoints: $BS / $MAX_POINTS"
echo "Aug/lr:          $AUG_DEG / $LR"
echo ""

cmd=(
    python msecnet/train_v2.py
    "$LABELS"
    "$PCD_DIR"
    --soft
    --steps "$STEPS"
    --tail-steps "$TAIL_STEPS"
    --bs "$BS"
    --max-points "$MAX_POINTS"
    --aug-deg "$AUG_DEG"
    --lr "$LR"
    --wd "$WD"
    --sched onecycle
    --onecycle-pct-start 0.05
    --onecycle-div-factor 25
    --onecycle-final-div-factor 10000
    --val-every "$VAL_EVERY"
    --log-every "$LOG_EVERY"
    --vis-every "$VIS_EVERY"
    --vis-samples "$VIS_SAMPLES"
    --save-last
    --out "$OUT_DIR"
)

if [[ "$TAIL_STEPS" -gt 0 ]]; then
    cmd+=(
        --tail-lr "${TAIL_LR:-1e-5}"
        --tail-wd "${TAIL_WD:-0}"
        --tail-pct-start "${TAIL_PCT_START:-0.15}"
        --tail-div-factor "${TAIL_DIV_FACTOR:-10}"
        --tail-final-div-factor "${TAIL_FINAL_DIV_FACTOR:-10000}"
    )
fi

(
    export CUDA_VISIBLE_DEVICES="$GPU"
    "${cmd[@]}"
) 2>&1 | tee "$OUT_DIR/stdout.log"

echo ""
echo "=========================================="
echo "Training finished."
echo "Metrics: $OUT_DIR/metrics.csv"
echo "Best:    $OUT_DIR/best.pt"
echo "=========================================="
