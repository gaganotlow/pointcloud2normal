#!/usr/bin/env bash
# Build the fixed OBB-expanded, unlabeled test dataset from the 20260730 open captures.
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_DIR="$PROJECT_ROOT/raw_data/20260730_五辆车外盖_多角度_三分类/open"
OUT_DIR="$PROJECT_ROOT/data/msecnet_20260730_open_obb10_unlabeled_test"
OBB_MODEL="/data2/shendu/code/ruoyu/fuelcap_6dpose/models/inner_obb_clean_v11m_0129/best.pt"

exec conda run --no-capture-output -n point2normal python \
  "$PROJECT_ROOT/shell/prepare_20260730_open_obb10_test.py" \
  "$SOURCE_DIR" "$OUT_DIR" \
  --obb-model "$OBB_MODEL" \
  --class-id 0 \
  --conf 0.25 \
  --expand 1.10 \
  "$@"
