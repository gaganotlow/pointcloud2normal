#!/usr/bin/env bash
# Run MSECNet + MoGe RGB feature ablations.
#
# Default experiments:
#   full = projected feat_map max/mean pool + cls_token
#   map  = projected feat_map max/mean pool only
#   cls  = cls_token only
#
# Usage:
#   cd /data2/shendu/code/ruoyu/train_point2normal
#   bash msecnet_add_rgb/run_ablation.sh
#
# Common overrides:
#   DEVICES="0 1" bash msecnet_add_rgb/run_ablation.sh
#   MODES="map cls" bash msecnet_add_rgb/run_ablation.sh
#   SKIP_EXISTING=1 bash msecnet_add_rgb/run_ablation.sh
#   INCLUDE_BASELINE=1 bash msecnet_add_rgb/run_ablation.sh

set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

LABELS="${LABELS:-shared/normal_labels_patch03.npz}"
PCD_DIR="${PCD_DIR:-data/pcd_dataset_roi}"
MOGE_FEAT_DIR="${MOGE_FEAT_DIR:-data/moge_features}"
OUT_ROOT="${OUT_ROOT:-msecnet_add_rgb}"

DEVICES="${DEVICES:-0 1}"
MODES="${MODES:-full map cls}"

STEPS="${STEPS:-12000}"
BS="${BS:-12}"
MAX_POINTS="${MAX_POINTS:-0}"
AUG_DEG="${AUG_DEG:-0}"
LR="${LR:-5e-4}"
VAL_EVERY="${VAL_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-100}"
VIS_EVERY="${VIS_EVERY:-1000}"
VIS_SAMPLES="${VIS_SAMPLES:-6}"
SOFT="${SOFT:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# Optional pure-geometry MSECNet baseline. Note this baseline script samples
# --npoints fixed points, while RGB fusion uses --max-points 0 by default.
INCLUDE_BASELINE="${INCLUDE_BASELINE:-0}"
BASELINE_STEPS="${BASELINE_STEPS:-12000}"
BASELINE_BS="${BASELINE_BS:-24}"
BASELINE_NPOINTS="${BASELINE_NPOINTS:-1024}"
BASELINE_AUG_DEG="${BASELINE_AUG_DEG:-0}"
BASELINE_LR="${BASELINE_LR:-5e-4}"
BASELINE_VAL_EVERY="${BASELINE_VAL_EVERY:-1000}"

declare -a DEVICE_LIST=()
declare -a MODE_LIST=()
read -r -a DEVICE_LIST <<< "$DEVICES"
read -r -a MODE_LIST <<< "$MODES"

if [[ ${#DEVICE_LIST[@]} -eq 0 ]]; then
    echo "ERROR: DEVICES is empty. Example: DEVICES=\"0 1\" bash $0" >&2
    exit 1
fi

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

feat_count="$(find "$MOGE_FEAT_DIR" -maxdepth 1 -name '*.npz' ! -name '*.tmp.npz' | wc -l)"
if [[ "$feat_count" -eq 0 ]]; then
    echo "ERROR: no MoGe feature .npz files found in $MOGE_FEAT_DIR" >&2
    echo "Run msecnet_add_rgb/precompute_moge_feat.py first." >&2
    exit 1
fi

echo "=========================================="
echo "MSECNet + MoGe RGB ablations"
echo "=========================================="
echo "ROOT:          $ROOT"
echo "LABELS:        $LABELS"
echo "PCD_DIR:       $PCD_DIR"
echo "MOGE_FEAT_DIR: $MOGE_FEAT_DIR ($feat_count npz files)"
echo "DEVICES:       ${DEVICE_LIST[*]}"
echo "MODES:         ${MODE_LIST[*]}"
echo "STEPS/BS:      $STEPS / $BS"
echo "OUT_ROOT:      $OUT_ROOT"
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

wait_all() {
    local status=0
    local i
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "[DONE] ${NAMES[$i]}"
        else
            echo "[FAIL] ${NAMES[$i]}" >&2
            status=1
        fi
    done
    PIDS=()
    NAMES=()
    if [[ "$status" -ne 0 ]]; then
        exit "$status"
    fi
}

launch_rgb_mode() {
    local mode="$1"
    local device="$2"
    local out_dir="$OUT_ROOT/ckpt_ablate_$mode"
    local stdout_log="$out_dir/stdout.log"

    if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/best.pt" ]]; then
        echo "[SKIP] rgb_mode=$mode because $out_dir/best.pt exists"
        return
    fi

    mkdir -p "$out_dir"
    echo "[LAUNCH] rgb_mode=$mode gpu=$device out=$out_dir"

    local -a cmd=(
        python msecnet_add_rgb/train_rgb_fusion.py
        "$LABELS"
        "$PCD_DIR"
        "$MOGE_FEAT_DIR"
        --steps "$STEPS"
        --bs "$BS"
        --max-points "$MAX_POINTS"
        --aug-deg "$AUG_DEG"
        --rgb-mode "$mode"
        --lr "$LR"
        --val-every "$VAL_EVERY"
        --log-every "$LOG_EVERY"
        --vis-every "$VIS_EVERY"
        --vis-samples "$VIS_SAMPLES"
        --out "$out_dir"
    )
    if [[ "$SOFT" == "1" ]]; then
        cmd+=(--soft)
    fi

    (
        export CUDA_VISIBLE_DEVICES="$device"
        "${cmd[@]}"
    ) > "$stdout_log" 2>&1 &

    PIDS+=("$!")
    NAMES+=("rgb_mode=$mode gpu=$device log=$stdout_log")
}

launch_baseline() {
    local device="$1"
    local out_dir="$OUT_ROOT/ckpt_ablate_geom"
    local stdout_log="$out_dir/stdout.log"

    if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/best.pt" ]]; then
        echo "[SKIP] pure geometry baseline because $out_dir/best.pt exists"
        return
    fi

    mkdir -p "$out_dir"
    echo "[LAUNCH] pure geometry baseline gpu=$device out=$out_dir"

    local -a cmd=(
        python msecnet/train_v1.py
        "$LABELS"
        "$PCD_DIR"
        --steps "$BASELINE_STEPS"
        --bs "$BASELINE_BS"
        --npoints "$BASELINE_NPOINTS"
        --aug-deg "$BASELINE_AUG_DEG"
        --lr "$BASELINE_LR"
        --val-every "$BASELINE_VAL_EVERY"
        --out "$out_dir"
    )
    if [[ "$SOFT" == "1" ]]; then
        cmd+=(--soft)
    fi

    (
        export CUDA_VISIBLE_DEVICES="$device"
        "${cmd[@]}"
    ) > "$stdout_log" 2>&1 &

    PIDS+=("$!")
    NAMES+=("pure_geometry gpu=$device log=$stdout_log")
}

device_i=0
for mode in "${MODE_LIST[@]}"; do
    case "$mode" in
        full|map|cls)
            launch_rgb_mode "$mode" "${DEVICE_LIST[$device_i]}"
            ;;
        *)
            echo "ERROR: unknown mode '$mode'. Use any subset of: full map cls" >&2
            exit 1
            ;;
    esac

    device_i=$((device_i + 1))
    if [[ "$device_i" -ge "${#DEVICE_LIST[@]}" ]]; then
        wait_all
        device_i=0
    fi
done
wait_all

if [[ "$INCLUDE_BASELINE" == "1" ]]; then
    launch_baseline "${DEVICE_LIST[0]}"
    wait_all
fi

echo ""
echo "=========================================="
echo "All requested experiments finished."
echo "Summary files:"
for mode in "${MODE_LIST[@]}"; do
    echo "  $OUT_ROOT/ckpt_ablate_$mode/metrics.csv"
    echo "  $OUT_ROOT/ckpt_ablate_$mode/best.pt"
done
if [[ "$INCLUDE_BASELINE" == "1" ]]; then
    echo "  $OUT_ROOT/ckpt_ablate_geom/metrics.csv"
    echo "  $OUT_ROOT/ckpt_ablate_geom/best.pt"
fi
echo "=========================================="
