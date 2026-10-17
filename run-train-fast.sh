#!/bin/bash
# SPEED MODE: H100 ($1.99/hr Lambda). Bigger batch, no grad-checkpointing, ~2x A100.
# Usage: bash run-train-fast.sh [train.jsonl] [ckpt-out] [epochs]
set -e
cd "$(dirname "$0")"
TRAIN=${1:-./data/train.jsonl}
OUT=${2:-./ckpt-fast}
EPOCHS=${3:-4}
python3 train_lora.py --train "$TRAIN" --out "$OUT" --epochs "$EPOCHS" --batch 8 --grad-ckpt 0
