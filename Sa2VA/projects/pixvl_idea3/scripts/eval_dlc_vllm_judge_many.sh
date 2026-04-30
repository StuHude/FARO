#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <output_dir> <name=/abs/path/pred.json> [name=/abs/path/pred.json ...]" >&2
  exit 1
fi

OUT="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON_BIN:-/dev/shm/xiaoyicheng_local/pixvl_idea1_fa/bin/python}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
if [[ -n "${EVAL_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${EVAL_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"
fi

JUDGE_MODEL=/mnt/pfs/xiaoyicheng/models/Meta-Llama-3.1-8B-Instruct
VLLM_PORT="${VLLM_PORT:-19100}"
BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 "$PY" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --model "$JUDGE_MODEL" \
  --served-model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --trust-remote-code \
  > "$OUT/vllm_server.log" 2>&1 &
VLLM_PID=$!

cleanup() {
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 120); do
  if curl -s "$BASE_URL/models" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if ! curl -s "$BASE_URL/models" >/dev/null 2>&1; then
  echo "vllm server failed to start" >&2
  exit 1
fi

ARGS=()
for item in "$@"; do
  ARGS+=(--item "$item")
done

CUDA_VISIBLE_DEVICES=0 "$PY" /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/scripts/eval_dlc_many_with_single_judge.py \
  --judge-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --device cuda:0 \
  --output-dir "$OUT" \
  "${ARGS[@]}"
