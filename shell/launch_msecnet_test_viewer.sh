#!/usr/bin/env bash
# Serve the web UI that lets users select a checkpoint, dataset, and split.
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PORT=${1:-8765}

cd "$PROJECT_ROOT"
echo "Open http://127.0.0.1:$PORT"
exec conda run --no-capture-output -n point2normal python web_label/server.py \
  --port "$PORT" \
  --msecnet-ui
