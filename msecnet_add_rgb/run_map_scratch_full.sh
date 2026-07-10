#!/usr/bin/env bash
# Full scratch training for the best ablation setting: rgb_mode=map.
#
# This does not load any previous checkpoint. It starts from random
# initialization and runs one continuous training job:
#   phase 1: original successful OneCycle recipe, 12000 steps
#   phase 2: low-LR OneCycle tail, 6000 steps
#
# Rationale:
#   - Original map run reached its best at the final 12000th step.
#   - Conservative low-LR OneCycle refinement improved map from 2.2559 deg to
#     2.1327 deg, but that used the old map checkpoint as initialization.
#   - This script folds that refinement into a single scratch run.
#
# Usage:
#   cd /data2/shendu/code/ruoyu/train_point2normal
#   bash msecnet_add_rgb/run_map_scratch_full.sh

set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

LABELS="${LABELS:-shared/normal_labels_patch03.npz}"
PCD_DIR="${PCD_DIR:-data/pcd_dataset_roi}"
MOGE_FEAT_DIR="${MOGE_FEAT_DIR:-data/moge_features}"
OUT_DIR="${OUT_DIR:-msecnet_add_rgb/ckpt_map_scratch_onecycle12k_tail6k}"
GPU="${GPU:-0}"

STEPS="${STEPS:-12000}"
TAIL_STEPS="${TAIL_STEPS:-6000}"
BS="${BS:-12}"
MAX_POINTS="${MAX_POINTS:-0}"
AUG_DEG="${AUG_DEG:-0}"
LR="${LR:-5e-4}"
WD="${WD:-1e-4}"
TAIL_LR="${TAIL_LR:-1e-5}"
TAIL_WD="${TAIL_WD:-0}"
VAL_EVERY="${VAL_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-100}"
VIS_EVERY="${VIS_EVERY:-2000}"
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
check_path "$MOGE_FEAT_DIR" "MoGe feature directory"

mkdir -p "$OUT_DIR"

echo "=========================================="
echo "Full scratch MSECNet + MoGe map training"
echo "=========================================="
echo "GPU:             $GPU"
echo "Output:          $OUT_DIR"
echo "Main steps/lr:   $STEPS / $LR"
echo "Tail steps/lr:   $TAIL_STEPS / $TAIL_LR"
echo "Batch/maxpoints: $BS / $MAX_POINTS"
echo "No init checkpoint is used."
echo ""

(
    export CUDA_VISIBLE_DEVICES="$GPU"
    python msecnet_add_rgb/train_rgb_fusion.py \
        "$LABELS" "$PCD_DIR" "$MOGE_FEAT_DIR" \
        --soft \
        --rgb-mode map \
        --steps "$STEPS" \
        --tail-steps "$TAIL_STEPS" \
        --tail-lr "$TAIL_LR" \
        --tail-wd "$TAIL_WD" \
        --tail-pct-start 0.15 \
        --tail-div-factor 10 \
        --tail-final-div-factor 10000 \
        --bs "$BS" \
        --max-points "$MAX_POINTS" \
        --aug-deg "$AUG_DEG" \
        --lr "$LR" \
        --wd "$WD" \
        --sched onecycle \
        --onecycle-pct-start 0.05 \
        --onecycle-div-factor 25 \
        --onecycle-final-div-factor 10000 \
        --val-every "$VAL_EVERY" \
        --log-every "$LOG_EVERY" \
        --vis-every "$VIS_EVERY" \
        --vis-samples "$VIS_SAMPLES" \
        --save-last \
        --out "$OUT_DIR"
) 2>&1 | tee "$OUT_DIR/stdout.log"

echo ""
echo "=========================================="
echo "Scratch training finished."
echo "Metrics: $OUT_DIR/metrics.csv"
echo "Best:    $OUT_DIR/best.pt"
echo "=========================================="
