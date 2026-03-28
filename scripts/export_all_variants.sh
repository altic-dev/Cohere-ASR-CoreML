#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SCRIPT="scripts/export_coreml_pure_pipeline.py"
VENV=".venv/bin/activate"

source "$VENV"

# We already have:
#   artifacts_fp16/  → --precision float16 --quantize none  (baseline)
#   artifacts/       → --precision float16 --quantize int8

# Export new variants:

echo "=== Exporting palettize4 (4-bit palettization) ==="
python3 "$SCRIPT" \
  --precision float16 \
  --quantize palettize4 \
  --artifacts-dir artifacts_palettize4

echo ""
echo "=== Exporting palettize6 (6-bit palettization) ==="
python3 "$SCRIPT" \
  --precision float16 \
  --quantize palettize6 \
  --artifacts-dir artifacts_palettize6

echo ""
echo "=== Exporting palettize4_int8lut (W4A8 joint compression) ==="
python3 "$SCRIPT" \
  --precision float16 \
  --quantize palettize4_int8lut \
  --artifacts-dir artifacts_w4a8

echo ""
echo "All variants exported."
