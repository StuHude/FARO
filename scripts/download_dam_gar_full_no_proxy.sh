#!/usr/bin/env bash

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/pfs/xiaoyicheng/data/pixvl_idea1}"
LOG_DIR="${LOG_DIR:-/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/logs}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python}"

mkdir -p "$DATA_ROOT/hf/dam" "$DATA_ROOT/hf/gar" "$LOG_DIR"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/pfs/xiaoyicheng/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

if command -v proxy_off >/dev/null 2>&1; then
  proxy_off || true
fi

"$PYTHON_BIN" - <<'PY'
import os
import time
from huggingface_hub import snapshot_download

tasks = [
    {
        "repo_id": "nvidia/describe-anything-dataset",
        "repo_type": "dataset",
        "local_dir": "/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dam",
    },
    {
        "repo_id": "HaochenWang/Grasp-Any-Region-Dataset",
        "repo_type": "dataset",
        "local_dir": "/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/gar",
    },
]

for task in tasks:
    repo_id = task["repo_id"]
    local_dir = task["local_dir"]
    print(f"[download] start {repo_id} -> {local_dir}", flush=True)
    last_error = None
    for attempt in range(1, 6):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=task["repo_type"],
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=1,
            )
            last_error = None
            print(f"[download] done {repo_id}", flush=True)
            break
        except Exception as exc:
            last_error = exc
            print(f"[download] attempt {attempt}/5 failed for {repo_id}: {exc}", flush=True)
            time.sleep(10 * attempt)
    if last_error is not None:
        raise last_error
PY
