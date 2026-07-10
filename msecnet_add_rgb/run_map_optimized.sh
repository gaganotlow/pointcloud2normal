#!/usr/bin/env bash
# Conservative follow-up training for the best ablation result: rgb_mode=map.
#
# Rationale from the ablation curves:
#   - map is currently best: 2.26 deg mean / 1.93 deg median.
#   - all validation bests happened at the final step, so 12k was a budget stop.
#   - The first aggressive cosine/EMA follow-up was worse: LR stayed too high
#     for too long, and EMA changed the original train/eval dynamics.
#
# This script launches two low-risk fine-tuning jobs initialized from the
# current map best checkpoint. Both validate step 0 first, so best.pt is never
# worse than the starting checkpoint.
#
# Usage:
#   cd /data2/shendu/code/ruoyu/train_point2normal
#   bash msecnet_add_rgb/run_map_optimized.sh

set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

LABELS="${LABELS:-shared/normal_labels_patch03.npz}"
PCD_DIR="${PCD_DIR:-data/pcd_dataset_roi}"
MOGE_FEAT_DIR="${MOGE_FEAT_DIR:-data/moge_features}"

GPU_LONG="${GPU_LONG:-0}"
GPU_FT="${GPU_FT:-1}"

BS="${BS:-12}"
MAX_POINTS="${MAX_POINTS:-0}"
AUG_DEG="${AUG_DEG:-0}"
VAL_EVERY="${VAL_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-100}"
VIS_EVERY="${VIS_EVERY:-2000}"
VIS_SAMPLES="${VIS_SAMPLES:-6}"

LONG_OUT="${LONG_OUT:-msecnet_add_rgb/ckpt_map_refine_const2e6}"
FT_OUT="${FT_OUT:-msecnet_add_rgb/ckpt_map_refine_onecycle1e5}"
INIT_CKPT="${INIT_CKPT:-msecnet_add_rgb/ckpt_ablate_map/best.pt}"

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
check_path "$INIT_CKPT" "initial map checkpoint"

echo "=========================================="
echo "Conservative MSECNet + MoGe map refinement"
echo "=========================================="
echo "Const-LR GPU:      $GPU_LONG -> $LONG_OUT"
echo "Low OneCycle GPU:  $GPU_FT -> $FT_OUT"
echo "Init checkpoint:   $INIT_CKPT"
echo "Batch/max-points:  $BS / $MAX_POINTS"
echo "Validation every:  $VAL_EVERY steps"
echo ""

declare -a PIDS=()
declare -a NAMES=()

cleanup() {
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo ""
        echo "Interrupted. Stopping running jobs..."
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done
    fi
}
trap cleanup INT TERM

mkdir -p "$LONG_OUT" "$FT_OUT"

echo "[LAUNCH] constant 2e-6 map refinement"
(
    export CUDA_VISIBLE_DEVICES="$GPU_LONG"
    python msecnet_add_rgb/train_rgb_fusion.py \
        "$LABELS" "$PCD_DIR" "$MOGE_FEAT_DIR" \
        --soft \
        --rgb-mode map \
        --init-ckpt "$INIT_CKPT" \
        --eval-at-start \
        --steps 4000 \
        --bs "$BS" \
        --max-points "$MAX_POINTS" \
        --aug-deg "$AUG_DEG" \
        --lr 2e-6 \
        --wd 0 \
        --sched constant \
        --patience 6 \
        --min-delta 0.002 \
        --val-every "$VAL_EVERY" \
        --log-every "$LOG_EVERY" \
        --vis-every "$VIS_EVERY" \
        --vis-samples "$VIS_SAMPLES" \
        --save-last \
        --out "$LONG_OUT"
) > "$LONG_OUT/stdout.log" 2>&1 &
PIDS+=("$!")
NAMES+=("constant 2e-6 map refinement log=$LONG_OUT/stdout.log")

echo "[LAUNCH] low OneCycle 1e-5 map refinement"
(
    export CUDA_VISIBLE_DEVICES="$GPU_FT"
    python msecnet_add_rgb/train_rgb_fusion.py \
        "$LABELS" "$PCD_DIR" "$MOGE_FEAT_DIR" \
        --soft \
        --rgb-mode map \
        --init-ckpt "$INIT_CKPT" \
        --eval-at-start \
        --steps 6000 \
        --bs "$BS" \
        --max-points "$MAX_POINTS" \
        --aug-deg "$AUG_DEG" \
        --lr 1e-5 \
        --wd 0 \
        --sched onecycle \
        --onecycle-pct-start 0.15 \
        --onecycle-div-factor 10 \
        --onecycle-final-div-factor 10000 \
        --patience 8 \
        --min-delta 0.002 \
        --val-every "$VAL_EVERY" \
        --log-every "$LOG_EVERY" \
        --vis-every "$VIS_EVERY" \
        --vis-samples "$VIS_SAMPLES" \
        --save-last \
        --out "$FT_OUT"
) > "$FT_OUT/stdout.log" 2>&1 &
PIDS+=("$!")
NAMES+=("low OneCycle 1e-5 map refinement log=$FT_OUT/stdout.log")

status=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[DONE] ${NAMES[$i]}"
    else
        echo "[FAIL] ${NAMES[$i]}" >&2
        status=1
    fi
done

if [[ "$status" -ne 0 ]]; then
    exit "$status"
fi

echo ""
echo "=========================================="
echo "Refinement runs finished."
echo "Compare:"
echo "  msecnet_add_rgb/ckpt_ablate_map/metrics.csv"
echo "  $LONG_OUT/metrics.csv"
echo "  $FT_OUT/metrics.csv"
echo "Best checkpoints:"
echo "  $LONG_OUT/best.pt"
echo "  $FT_OUT/best.pt"
echo "=========================================="
