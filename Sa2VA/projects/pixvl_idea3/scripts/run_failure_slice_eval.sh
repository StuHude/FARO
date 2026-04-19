#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
PYTHON_BIN="${PYTHON_BIN:-/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python}"

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/projects/pixvl_idea3/configs/idea3_mvp_routed_opd_rl.py}"
ADAPTER_PATH="${ADAPTER_PATH:-/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/stage3_routed_opd_rl/adapter}"
SCHEMA_ROOT="${SCHEMA_ROOT:-/mnt/pfs/xiaoyicheng/data/pixvl_idea3/schemas}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/eval}"

mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" -m projects.pixvl_idea1.eval.eval_dlc \
  --config "$CONFIG_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --schema-file "$SCHEMA_ROOT/semantic_slice_eval.jsonl" \
  --output "$OUTPUT_ROOT/semantic.json"

"$PYTHON_BIN" -m projects.pixvl_idea1.eval.eval_refseg \
  --config "$CONFIG_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --schema-file "$SCHEMA_ROOT/relation_slice_eval.jsonl" \
  --output "$OUTPUT_ROOT/relation.json"

"$PYTHON_BIN" -m projects.pixvl_idea1.eval.eval_refseg \
  --config "$CONFIG_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --schema-file "$SCHEMA_ROOT/geometry_slice_eval.jsonl" \
  --output "$OUTPUT_ROOT/geometry.json"

"$PYTHON_BIN" -m projects.pixvl_idea1.eval.eval_refseg \
  --config "$CONFIG_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --schema-file "$SCHEMA_ROOT/refseg_val_routed.jsonl" \
  --output "$OUTPUT_ROOT/refseg_overall.json"

"$PYTHON_BIN" -m projects.pixvl_idea1.eval.eval_dlc \
  --config "$CONFIG_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --schema-file "$SCHEMA_ROOT/dlc_eval.jsonl" \
  --output "$OUTPUT_ROOT/maskcap_overall.json"

"$PYTHON_BIN" -m projects.pixvl_idea3.eval.summarize_failure_slices \
  --semantic "$OUTPUT_ROOT/semantic.json" \
  --relation "$OUTPUT_ROOT/relation.json" \
  --geometry "$OUTPUT_ROOT/geometry.json" \
  --refseg-overall "$OUTPUT_ROOT/refseg_overall.json" \
  --maskcap-overall "$OUTPUT_ROOT/maskcap_overall.json" \
  --output "$OUTPUT_ROOT/summary.json"
