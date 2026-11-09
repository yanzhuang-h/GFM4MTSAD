#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${ENTITY:?Set ENTITY to SMD_1-1, SMAP_T-3, or MSL_P-15}"
: "${BACKEND:?Set BACKEND to anomalygfm or owleye}"
: "${OUTPUT:?Set OUTPUT to a new run directory}"
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
exec "${PYTHON:-python}" -u main.py run \
  --entity "$ENTITY" --backend "$BACKEND" --seed "${SEED:-0}" \
  --assets "${ASSETS:-configs/assets.local.json}" \
  --threads "${THREADS:-2}" --output "$OUTPUT"
