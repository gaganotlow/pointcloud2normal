#!/usr/bin/env bash
# Evaluate the 9211 held-out split, then serve it in the read-only web_label viewer.
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CHECKPOINT=${1:-"$PROJECT_ROOT/msecnet_best/out/pseudo_obb_9211_legacy_v1/best.pt"}
PORT=${2:-8765}
DATASET_DIR="$PROJECT_ROOT/data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb"
SOURCE_ROOT="$PROJECT_ROOT/data/fuelcap_pass_20260721_9211"
OUTPUT_DIR="$(dirname -- "$CHECKPOINT")/inference_test"

cd "$PROJECT_ROOT"
conda run --no-capture-output -n point2normal python msecnet_best/infer.py \
  "$CHECKPOINT" \
  "$DATASET_DIR/labels_manual3d.npz" \
  "$DATASET_DIR/clouds" \
  --centers "$DATASET_DIR/anchors_manual3d.json" \
  --split "$DATASET_DIR/split_by_car_model.json" \
  --split-name test \
  --out "$OUTPUT_DIR"

echo "Open http://127.0.0.1:$PORT"
exec conda run --no-capture-output -n point2normal python web_label/server.py \
  --port "$PORT" \
  --msecnet-report "$OUTPUT_DIR/report.json" \
  --msecnet-dataset "$DATASET_DIR" \
  --msecnet-source-root "$SOURCE_ROOT"
