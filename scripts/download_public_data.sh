#!/usr/bin/env bash

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/pfs/xiaoyicheng/data/pixvl_idea1}"
MODE="${1:---smoke}"
export DATA_ROOT
export MODE

mkdir -p "$DATA_ROOT"/{hf,raw,eval,schemas,smoke}

export HF_HOME="${HF_HOME:-/mnt/pfs/xiaoyicheng/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

if ! command -v python >/dev/null 2>&1; then
  echo "python 未安装，无法下载数据。"
  exit 1
fi

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

python - <<'PY'
import os
import time
from huggingface_hub import snapshot_download

data_root = os.environ.get("DATA_ROOT", "/mnt/pfs/xiaoyicheng/data/pixvl_idea1")
mode = os.environ.get("MODE", "--smoke")

downloads = [
    {
        "repo_id": "Dense-World/Sa2VA-Training",
        "repo_type": "dataset",
        "local_dir": os.path.join(data_root, "hf", "sa2va_training"),
        "allow_patterns": ["README.md"],
    },
    {
        "repo_id": "bitersun/Sa2VA-finetune-example",
        "repo_type": "dataset",
        "local_dir": os.path.join(data_root, "hf", "sa2va_finetune_example"),
        "allow_patterns": ["*"],
    },
    {
        "repo_id": "nvidia/DLC-Bench",
        "repo_type": "dataset",
        "local_dir": os.path.join(data_root, "hf", "dlc_bench"),
        "allow_patterns": [
            "*",
        ] if mode != "--smoke" else [
            "README.md",
            "*.json",
            "*.parquet",
            "*.txt",
        ],
    },
]

if mode != "--smoke":
    downloads.extend(
        [
            {
                "repo_id": "HaochenWang/Grasp-Any-Region-Dataset",
                "repo_type": "dataset",
                "local_dir": os.path.join(data_root, "hf", "gar"),
                "allow_patterns": [
                    "README.md",
                    ".gitattributes",
                    "Fine-Grained-Dataset-Part1/**",
                ],
            },
            {
                "repo_id": "nvidia/describe-anything-dataset",
                "repo_type": "dataset",
                "local_dir": os.path.join(data_root, "hf", "dam"),
                "allow_patterns": [
                    "README.md",
                    ".gitattributes",
                    "COCOStuff/**",
                    "LVIS/**",
                    "PACO/**",
                ],
            },
        ]
    )

for item in downloads:
    print(f"Downloading {item['repo_id']} -> {item['local_dir']}")
    last_error = None
    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=item["repo_id"],
                repo_type=item["repo_type"],
                local_dir=item["local_dir"],
                local_dir_use_symlinks=False,
                allow_patterns=item["allow_patterns"],
                resume_download=True,
                max_workers=1,
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            print(f"[attempt {attempt}/3] failed for {item['repo_id']}: {exc}")
            time.sleep(3 * attempt)
    if last_error is not None:
        raise last_error
PY

if [[ "$MODE" != "--smoke" ]]; then
  mkdir -p "$DATA_ROOT/raw/ref_seg"
  mkdir -p "$DATA_ROOT/hf/sa2va_training"
  for zip_name in ref_seg_coco.zip ref_seg_coco_plus.zip ref_seg_coco_g.zip; do
    while true; do
      if download_file "$HF_ENDPOINT/datasets/Dense-World/Sa2VA-Training/resolve/main/$zip_name" "$DATA_ROOT/hf/sa2va_training/$zip_name"; then
        break
      fi
      sleep 5
    done
  done
  for zip_name in ref_seg_coco.zip ref_seg_coco_plus.zip ref_seg_coco_g.zip; do
    zip_path="$DATA_ROOT/hf/sa2va_training/$zip_name"
    if [[ -f "$zip_path" ]]; then
      unzip -n "$zip_path" -d "$DATA_ROOT/raw/ref_seg"
    fi
  done
fi

echo "公开数据下载完成: $DATA_ROOT"
