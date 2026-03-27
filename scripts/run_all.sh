#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/.venv/bin/activate"

python "${ROOT_DIR}/scripts/run_baseline.py"
python "${ROOT_DIR}/scripts/export_coreml.py" --quantize
python "${ROOT_DIR}/scripts/test_coreml_python.py"
python "${ROOT_DIR}/scripts/export_coreml_fullseq.py"

pushd "${ROOT_DIR}/swift_runner" >/dev/null
swift run swift_runner \
  "${ROOT_DIR}/artifacts/cohere_first_step.mlpackage" \
  "${ROOT_DIR}/artifacts/swift_input.json"
swift run swift_fullseq_runner \
  "${ROOT_DIR}/artifacts/cohere_decoder_fullseq.mlpackage" \
  "${ROOT_DIR}/artifacts/swift_fullseq_input.json" \
  96
if [[ "${RUN_CACHED_EXPERIMENTAL:-0}" == "1" ]]; then
  python "${ROOT_DIR}/scripts/export_coreml_cached.py"
  swift run swift_cached_runner \
    "${ROOT_DIR}/artifacts/cohere_decoder_cached.mlpackage" \
    "${ROOT_DIR}/artifacts/swift_cached_input.json" \
    96
fi
popd >/dev/null
