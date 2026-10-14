#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    exec "${PYTHON_BIN}" "${APP_ROOT}/server.py" "$@"
fi
exec conda run --no-capture-output -n "${CONDA_ENV:-da3-ros}" python "${APP_ROOT}/server.py" "$@"
