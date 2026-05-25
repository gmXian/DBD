#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/yneversky/data}
ARCH=${ARCH:-ViT-B/32}
LT_SPLIT=${LT_SPLIT:-val}
ATTACK=${ATTACK:-pgd}
EPS=${EPS:-4.0}
ALPHA=${ALPHA:-1.0}
STEPS=${STEPS:-100}
SEED=${SEED:-0}
NUM_WORKERS=${NUM_WORKERS:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
OUTPUT_ROOT=${OUTPUT_ROOT:-./adv_images}

python generate_adv_images.py \
  --test_sets ImageNetLT \
  --data_root "${DATA_ROOT}" \
  --lt_split "${LT_SPLIT}" \
  --arch "${ARCH}" \
  --attack "${ATTACK}" \
  --eps "${EPS}" \
  --alpha "${ALPHA}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  -n "${NUM_WORKERS}" \
  -b "${BATCH_SIZE}" \
  --output_root "${OUTPUT_ROOT}"
