#!/usr/bin/env bash
# Build a clean, uploadable source bundle of the repo (no weights, no generated data).
#
#   ./package.sh                    -> dist/tradesman-v1-src.tar.gz  (+ .zip)
#
# Weights are NOT included on purpose: model.safetensors is 1.7 GB and GitHub
# rejects files over 100 MB (LFS free tier is 1 GB total). Ship them separately —
# see PACKAGING.md.
set -euo pipefail
cd "$(dirname "$0")"

OUT=dist
NAME=tradesman-v1-src
rm -rf "$OUT/$NAME" "$OUT/$NAME.tar.gz" "$OUT/$NAME.zip"
mkdir -p "$OUT/$NAME"

rsync -a \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='ckpt*/' \
  --exclude='*.log' \
  --exclude='synth/' \
  --exclude='blueprints/all/' \
  --exclude='data/*.jsonl' \
  --exclude='uploads/' \
  --exclude='test/' \
  --exclude='.harness/' \
  --exclude='dist/' \
  ./ "$OUT/$NAME/"

# ship the generated training/val jsonl in the bundle (needed to reproduce training,
# gitignored in the repo because it is derived data)
[ -f data/train_v1.jsonl ] && cp data/train_v1.jsonl "$OUT/$NAME/data/train_v1.jsonl"

( cd "$OUT" && tar czf "$NAME.tar.gz" "$NAME" )
( cd "$OUT" && zip -qr "$NAME.zip" "$NAME" )

echo "bundle: $OUT/$NAME.tar.gz"
du -sh "$OUT/$NAME.tar.gz" "$OUT/$NAME.zip"
