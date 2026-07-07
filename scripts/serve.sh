#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m http.server "${PORT:-8027}" --bind "${HOST:-0.0.0.0}"

