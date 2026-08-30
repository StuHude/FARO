#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
MODEL=${SAMTOK_BASE_CHECKPOINT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co}
DATA=${SAMTOK_SELECTIVE_DATA:-$FARO_ROOT/data/fepo_existence/egfepo_train_5120.jsonl}
ADAPTER=${SAMTOK_STANDALONE_ADAPTER:-$FARO_ROOT/outputs/samtok_selective/continued_sft_to500/adapter}
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/samtok_selective/configs/continued_sft_r18_matched_200.py}
JOB_NAME=${JOB_NAME:-dna-fepo-r18-matched-sft-200-2g}
TAGS_FILE=$FARO_ROOT/rjob_tags.txt
[[ -f "$CONFIG" && -f "$DATA" && -f "$TAGS_FILE" ]] || { echo "Missing matched SFT input" >&2; exit 2; }
rows=$(awk 'NF {n++} END {print n+0}' "$DATA")
(( rows >= 5000 )) || { echo "Matched SFT requires at least 5000 rows, got $rows" >&2; exit 2; }
case "$JOB_NAME" in dna-*) ;; *) echo "JOB_NAME must start with dna-" >&2; exit 2 ;; esac
export FARO_ROOT SAMTOK_BASE_CHECKPOINT="$MODEL" SAMTOK_SELECTIVE_DATA="$DATA" SAMTOK_STANDALONE_ADAPTER="$ADAPTER" CONFIG JOB_NAME GPU_COUNT=2 TAGS_FILE
exec bash "$SCRIPT_DIR/submit_samtok_standalone_sft.sh"
