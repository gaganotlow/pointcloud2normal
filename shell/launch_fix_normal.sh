#!/usr/bin/env bash
# Serve the train/validation/test normal-label repair UI.
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PORT=${1:-8766}

cd "$PROJECT_ROOT"
echo "Open http://127.0.0.1:$PORT"
exec conda run --no-capture-output -n point2normal python web_label/app_fix_normal.py \
  --port "$PORT"
