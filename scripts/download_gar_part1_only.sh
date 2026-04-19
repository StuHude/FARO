#!/usr/bin/env bash

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/pfs/xiaoyicheng/data/pixvl_idea1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

download_file() {
  local url="$1"
  local output="$2"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --continue=true \
      --max-connection-per-server=16 \
      --split=16 \
      --min-split-size=1M \
      --max-tries=0 \
      --timeout=120 \
      --file-allocation=none \
      --summary-interval=0 \
      --dir="$(dirname "$output")" \
      --out="$(basename "$output")" \
      "$url"
  else
    wget -c --retry-connrefused --waitretry=5 --read-timeout=120 --timeout=120 \
      -O "$output" \
      "$url"
  fi
}

TARGET_DIR="$DATA_ROOT/hf/gar/Fine-Grained-Dataset-Part1"
mkdir -p "$TARGET_DIR"

for file in README.md .gitattributes; do
  download_file "$HF_ENDPOINT/datasets/HaochenWang/Grasp-Any-Region-Dataset/resolve/main/$file" "$DATA_ROOT/hf/gar/$file"
done

for idx in $(seq -f "%05g" 0 43); do
  file="data-${idx}-of-00044.arrow"
  while true; do
    if download_file \
      "$HF_ENDPOINT/datasets/HaochenWang/Grasp-Any-Region-Dataset/resolve/main/Fine-Grained-Dataset-Part1/$file" \
      "$TARGET_DIR/$file"; then
      break
    fi
    sleep 5
  done
done
