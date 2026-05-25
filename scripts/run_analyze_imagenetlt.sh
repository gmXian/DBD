#!/usr/bin/env bash
set -euo pipefail

LT_SPLIT=${LT_SPLIT:-val}
ARCH=${ARCH:-ViT-B/32}
ATTACK=${ATTACK:-pgd}
EPS=${EPS:-4.0}
STEPS=${STEPS:-100}
OUT_DIR=${OUT_DIR:-outputs/imagenetlt}
SPLIT_CSV=${SPLIT_CSV:-data/ImageNet_LT_class_splits.csv}
ARCH_TAG=${ARCH//\//-}

python tools/analyze_lt_results.py \
  --clean_csv "${OUT_DIR}/clean_${LT_SPLIT}_${ARCH_TAG}_classwise.csv" \
  --adv_csv "${OUT_DIR}/adv_${LT_SPLIT}_${ARCH_TAG}_${ATTACK}_steps${STEPS}_eps${EPS}_classwise.csv" \
  --split_csv "${SPLIT_CSV}" \
  --out_dir "${OUT_DIR}/summary_${LT_SPLIT}_${ARCH_TAG}_${ATTACK}_steps${STEPS}_eps${EPS}"
