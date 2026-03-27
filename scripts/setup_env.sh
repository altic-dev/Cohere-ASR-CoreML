#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

echo "[setup] root=${ROOT_DIR}"
echo "[setup] python=${PYTHON_BIN}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "${ROOT_DIR}/requirements.txt"

python - <<'PY'
import importlib
mods = ["torch", "transformers", "huggingface_hub", "coremltools", "soundfile", "numpy", "sentencepiece"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing deps after install: {missing}")
print("Dependency check passed.")
PY

echo "[setup] complete"
