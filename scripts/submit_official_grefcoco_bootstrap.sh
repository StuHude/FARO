#!/usr/bin/env bash
set -euo pipefail

FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
ROOT=${ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab}
JOB_NAME=${JOB_NAME:-dna-official-grefcoco-bootstrap-20k}
LEFT=${LEFT:-$FARO_ROOT/evals/official_grefcoco_base_full/merged}
RIGHT=${RIGHT:-$FARO_ROOT/evals/official_grefcoco_continued_sft_full/merged}
OUTPUT=${OUTPUT:-$FARO_ROOT/evals/official_grefcoco_base_vs_contsft_bootstrap20k.json}
TAGS_FILE=${TAGS_FILE:-$FARO_ROOT/rjob_tags.txt}

case "$JOB_NAME" in dna-*) ;; *) echo "JOB_NAME must start with dna-" >&2; exit 2 ;; esac
[[ -d "$LEFT" && -d "$RIGHT" ]] || { echo "Missing paired GRefCOCO outputs" >&2; exit 2; }
[[ -f "$TAGS_FILE" ]] || { echo "Missing positive-tag file: $TAGS_FILE" >&2; exit 2; }
POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }

rjob submit --name="$JOB_NAME" --namespace=ailab-dnacoding \
  --cpu=16 --gpu=1 --memory=64000 --positive-tags="$POSITIVE_TAGS" \
  --charged-group=dnacoding_gpu --private-machine=group \
  --mount=gpfs://gpfs1/dnacoding:/mnt/shared-storage-user/dnacoding \
  --mount=gpfs://gpfs1/wuyucheng:/mnt/shared-storage-user/wuyucheng \
  --image=registry.h.pjlab.org.cn/ailab-dnacoding/wuyucheng:test1 \
  --custom-resources=brainpp.cn/fuse=1 --enable-sshd -- bash -lc "
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
cd /opt; mkdir -p /opt/vlm
if [ -f /opt/vlm_env.tar.gz ]; then tar -xzf /opt/vlm_env.tar.gz -C /opt/vlm; rm -f /opt/vlm_env.tar.gz; elif [ -f vlm_env.tar.gz ]; then tar -xzf vlm_env.tar.gz -C /opt/vlm; rm -f vlm_env.tar.gz; fi
/opt/vlm/bin/python /opt/vlm/bin/conda-unpack || true
mkdir -p '$(dirname "$OUTPUT")'
/opt/vlm/bin/python '$FARO_ROOT/tools/analyze_official_grefcoco_pair.py' '$LEFT' '$RIGHT' \
  --repeats 20000 --seed 42 --output '$OUTPUT'
"
