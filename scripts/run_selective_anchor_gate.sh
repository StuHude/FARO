#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'MSG'
Historical Idea3 selective-anchor gate disabled. It writes/reads the retired
PixVL experiment tree; use the SAMTok-only adaptive evaluator and registered
FARO outputs instead.
MSG
exit 2

ROOT=${ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab}
FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
OUT_ROOT=${OUT_ROOT:-$ROOT/outputs/pixvl_idea3}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/evals}
SCHEMA=${SCHEMA:-$FARO_ROOT/data/fepo_existence/grefcoco_selective_holdout_256.jsonl}
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/pixvl_idea3/configs/samtok_selective_refseg_eval.py}
TRAIN_POLL_SECONDS=${TRAIN_POLL_SECONDS:-30}
EVAL_POLL_SECONDS=${EVAL_POLL_SECONDS:-300}
TRAIN_TIMEOUT_SECONDS=${TRAIN_TIMEOUT_SECONDS:-14400}
EVAL_TIMEOUT_SECONDS=${EVAL_TIMEOUT_SECONDS:-14400}

declare -A adapters=(
  [anchor_rl]="$OUT_ROOT/samtok_selective_sft_anchor_rl_2gpu/adapter"
  [anchor_area_rl]="$OUT_ROOT/samtok_selective_sft_anchor_area_rl_2gpu/adapter"
  [continued_sft]="$OUT_ROOT/samtok_selective_sft_anchor_continue_sft_2gpu/adapter"
)
declare -A outputs=(
  [anchor_rl]="$EVAL_ROOT/samtok_selective_sft_anchor_rl_eval.json"
  [anchor_area_rl]="$EVAL_ROOT/samtok_selective_sft_anchor_area_rl_eval.json"
  [continued_sft]="$EVAL_ROOT/samtok_selective_sft_anchor_continue_sft_eval.json"
)

started=$(date +%s)
while true; do
  ready=1
  for key in "${!adapters[@]}"; do
    [[ -s "${adapters[$key]}/adapter_model.safetensors" ]] || ready=0
  done
  (( ready == 1 )) && break
  now=$(date +%s)
  if (( now - started >= TRAIN_TIMEOUT_SECONDS )); then
    echo "Timed out waiting for all three training adapters" >&2
    exit 1
  fi
  sleep "$TRAIN_POLL_SECONDS"
done

mkdir -p "$EVAL_ROOT" "$FARO_ROOT/codex_resume"
pids=()
for key in anchor_rl anchor_area_rl continued_sft; do
  rm -f "${outputs[$key]}"
  JOB_PREFIX="dna-samtok-${key//_/-}-eval" \
  ADAPTER="${adapters[$key]}" OUTPUT="${outputs[$key]}" CONFIG="$CONFIG" \
  REFSEG_SCHEMA="$SCHEMA" GEOMETRY_SCHEMA=none MASKCAP_SCHEMA=none EXISTENCE_SCHEMA=none \
  POLL_SECONDS="$EVAL_POLL_SECONDS" \
  bash "$FARO_ROOT/scripts/submit_fepo_eval_adaptive.sh" \
    > "$FARO_ROOT/codex_resume/${key}_eval_adaptive.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

started=$(date +%s)
while true; do
  ready=1
  for key in "${!outputs[@]}"; do
    [[ -s "${outputs[$key]}" ]] || ready=0
  done
  (( ready == 1 )) && break
  now=$(date +%s)
  if (( now - started >= EVAL_TIMEOUT_SECONDS )); then
    echo "Timed out waiting for all three 512-row evaluations" >&2
    exit 1
  fi
  sleep "$TRAIN_POLL_SECONDS"
done

echo "SELECTIVE_ANCHOR_GATE_EVALS_READY"
