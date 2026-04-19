#!/usr/bin/env bash

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/pfs/xiaoyicheng/data/pixvl_idea1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

download_file() {
  local url="$1"
  local output="$2"
  if [[ -f "$output" ]]; then
    if unzip -tq "$output" >/dev/null 2>&1; then
      rm -f "${output}.aria2"
      return 0
    fi
    rm -f "$output" "${output}.aria2"
  fi
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

mkdir -p "$DATA_ROOT/hf/sa2va_training" "$DATA_ROOT/raw/ref_seg"

for zip_name in ref_seg_coco.zip ref_seg_coco_plus.zip ref_seg_coco_g.zip; do
  while true; do
    if download_file "$HF_ENDPOINT/datasets/Dense-World/Sa2VA-Training/resolve/main/$zip_name" "$DATA_ROOT/hf/sa2va_training/$zip_name"; then
      break
    fi
    sleep 5
  done
  unzip -n "$DATA_ROOT/hf/sa2va_training/$zip_name" -d "$DATA_ROOT/raw/ref_seg"
done
