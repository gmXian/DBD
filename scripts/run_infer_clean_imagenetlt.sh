#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/yneversky/data}
ARCH=${ARCH:-ViT-B/32}
LT_SPLIT=${LT_SPLIT:-val}
SEED=${SEED:-0}
NUM_WORKERS=${NUM_WORKERS:-8}
OUT_DIR=${OUT_DIR:-outputs/imagenetlt}

mkdir -p "${OUT_DIR}"

python infer_dbd.py \
  --test_sets ImageNetLT \
  --data_root "${DATA_ROOT}" \
  --lt_split "${LT_SPLIT}" \
  --adv_dir clean \
  --arch "${ARCH}" \
  -n "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --classwise_csv "${OUT_DIR}/clean_${LT_SPLIT}_${ARCH//\//-}_classwise.csv"
