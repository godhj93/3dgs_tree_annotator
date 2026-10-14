#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/branch_detector/build"
if [[ "${NATIVE_ONLY:-0}" != "1" ]]; then
  clang++ --target=wasm32 -std=c++17 -O3 -nostdlib \
    -Wl,--no-entry -Wl,--export-all -Wl,--initial-memory=33554432 \
    -Wl,--max-memory=67108864 \
    "$ROOT/branch_detector/branch_detector.cpp" \
    -o "$ROOT/supersplat/static/lib/branch-detector/branch_detector.wasm"
fi
g++ -std=c++17 -O3 -Wall -Wextra \
  "$ROOT/branch_detector/branch_detector.cpp" "$ROOT/branch_detector/benchmark.cpp" \
  -o "$ROOT/branch_detector/build/benchmark"
