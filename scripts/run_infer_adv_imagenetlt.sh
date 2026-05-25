#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/yneversky/data}
ARCH=${ARCH:-ViT-B/32}
LT_SPLIT=${LT_SPLIT:-val}
ATTACK=${ATTACK:-pgd}
EPS=${EPS:-4.0}
STEPS=${STEPS:-10}
SEED=${SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-8}
OUT_DIR=${OUT_DIR:-outputs/imagenetlt}
ADV_ROOT=${ADV_ROOT:-./adv_images}
ARCH_TAG=${ARCH//\//-}
ADV_DIR=${ADV_DIR:-${ADV_ROOT}/${ARCH_TAG}_${ATTACK}_steps${STEPS}_eps${EPS}_${LT_SPLIT}}

mkdir -p "${OUT_DIR}"

python infer_dbd.py \
  --test_sets ImageNetLT \
  --data_root "${DATA_ROOT}" \
  --lt_split "${LT_SPLIT}" \
  --adv_dir "${ADV_DIR}" \
  --arch "${ARCH}" \
  -n "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --classwise_csv "${OUT_DIR}/adv_${LT_SPLIT}_${ARCH_TAG}_${ATTACK}_steps${STEPS}_eps${EPS}_classwise.csv"
