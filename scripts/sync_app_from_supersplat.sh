#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d supersplat/dist ]; then
  echo "supersplat/dist does not exist. Run npm run build inside supersplat first." >&2
  exit 1
fi

mkdir -p app
cp -a supersplat/dist/. app/

