#!/usr/bin/env bash

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/pfs/xiaoyicheng/data/pixvl_idea1}"
LOG_DIR="${LOG_DIR:-/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/logs}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REL_DIR="$DATA_ROOT/hf/gar/Relation-Dataset"
LIST_FILE="$LOG_DIR/gar_relation_aria2_input.txt"
ARIA2_JOBS="${ARIA2_JOBS:-8}"
ARIA2_SPLIT="${ARIA2_SPLIT:-8}"
ARIA2_CONN_PER_SERVER="${ARIA2_CONN_PER_SERVER:-8}"

mkdir -p "$REL_DIR" "$LOG_DIR"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY
if command -v proxy_off >/dev/null 2>&1; then
  proxy_off || true
fi

cat > "$LIST_FILE" <<EOF
$HF_ENDPOINT/datasets/HaochenWang/Grasp-Any-Region-Dataset/resolve/main/README.md
 dir=$DATA_ROOT/hf/gar
 out=README.md
$HF_ENDPOINT/datasets/HaochenWang/Grasp-Any-Region-Dataset/resolve/main/.gitattributes
 dir=$DATA_ROOT/hf/gar
 out=.gitattributes
EOF

for idx in $(seq -f "%05g" 0 544); do
  cat >> "$LIST_FILE" <<EOF
$HF_ENDPOINT/datasets/HaochenWang/Grasp-Any-Region-Dataset/resolve/main/Relation-Dataset/data-${idx}-of-00545.arrow
 dir=$REL_DIR
 out=data-${idx}-of-00545.arrow
EOF
done

aria2c \
  --continue=true \
  --allow-overwrite=false \
  --auto-file-renaming=false \
  --max-concurrent-downloads="$ARIA2_JOBS" \
  --max-connection-per-server="$ARIA2_CONN_PER_SERVER" \
  --split="$ARIA2_SPLIT" \
  --min-split-size=4M \
  --max-tries=0 \
  --retry-wait=5 \
  --timeout=120 \
  --file-allocation=none \
  --summary-interval=10 \
  --console-log-level=notice \
  --input-file="$LIST_FILE"
