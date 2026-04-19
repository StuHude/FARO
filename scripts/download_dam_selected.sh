#!/usr/bin/env bash

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "${ENV_PREFIX:-/mnt/pfs/xiaoyicheng/envs/pixvl_idea1}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

python /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/download_dam_selected.py \
  --repo-id "${1:-nvidia/describe-anything-dataset}" \
  --output-root "${2:-/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dam}" \
  --subset-names "${@:3}"
