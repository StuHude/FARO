#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/projects/pixvl_idea1/configs/idea1_joint_opd_rl.py}"
ADAPTER_PATH="${ADAPTER_PATH:-/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage3_joint_opd_rl/adapter}"
REFSEG_SCHEMA="${REFSEG_SCHEMA:-/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/refseg_val.jsonl}"
MASKCAP_SCHEMA="${MASKCAP_SCHEMA:-/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/dlc_bench_train.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/eval}"

mkdir -p "$OUTPUT_ROOT"

python -m projects.pixvl_idea1.eval.eval_refseg \
  --config "$CONFIG_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --schema-file "$REFSEG_SCHEMA" \
  --output "$OUTPUT_ROOT/refseg.json"

python -m projects.pixvl_idea1.eval.eval_dlc \
  --config "$CONFIG_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --schema-file "$MASKCAP_SCHEMA" \
  --output "$OUTPUT_ROOT/maskcap.json"

python -m projects.pixvl_idea1.eval.eval_joint_summary \
  --refseg "$OUTPUT_ROOT/refseg.json" \
  --maskcap "$OUTPUT_ROOT/maskcap.json" \
  --output "$OUTPUT_ROOT/summary.json"
