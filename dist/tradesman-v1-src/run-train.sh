#!/bin/bash
# Cheap training run. Defaults fit L4 24GB ($0.17/hr). Override for A100 burst.
# Usage: bash run-train.sh [train.jsonl] [ckpt-out]
set -e
cd "$(dirname "$0")"
TRAIN=${1:-./data/train.jsonl}
OUT=${2:-./ckpt}
python3 train_lora.py --train "$TRAIN" --out "$OUT" --epochs 4 --batch 2 --grad-ckpt 1
