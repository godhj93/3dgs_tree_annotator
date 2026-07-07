#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../supersplat"
npm ci
npm run build
cd ..
bash scripts/sync_app_from_supersplat.sh

