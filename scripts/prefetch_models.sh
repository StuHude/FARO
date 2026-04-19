#!/usr/bin/env bash

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "${ENV_PREFIX:-/mnt/pfs/xiaoyicheng/envs/pixvl_idea1}"

export HF_HOME="${HF_HOME:-/mnt/pfs/xiaoyicheng/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

REPO_ID="${1:-zhouyik/Qwen3-VL-4B-SAMTok-co}"
LOCAL_DIR="${2:-/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co}"
mkdir -p "$LOCAL_DIR"

download_file() {
  local url="$1"
  local output="$2"
  if [[ -f "$output" && ! -f "${output}.aria2" ]]; then
    return 0
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

if [[ "$REPO_ID" == "zhouyik/Qwen3-VL-4B-SAMTok-co" ]]; then
  FILES=(
    ".gitattributes"
    "added_tokens.json"
    "chat_template.jinja"
    "config.json"
    "generation_config.json"
    "mask_tokenizer_256x2.pth"
    "merges.txt"
    "model-00001-of-00002.safetensors"
    "model-00002-of-00002.safetensors"
    "model.safetensors.index.json"
    "processor_config.json"
    "sam2.1_hiera_large.pt"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
  )
  for file in "${FILES[@]}"; do
    while true; do
      if download_file "$HF_ENDPOINT/$REPO_ID/resolve/main/$file" "$LOCAL_DIR/$file"; then
        break
      fi
      sleep 5
    done
  done
else
  python /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/prefetch_models.py \
    --repo-id "$REPO_ID" \
    --local-dir "$LOCAL_DIR"
fi
