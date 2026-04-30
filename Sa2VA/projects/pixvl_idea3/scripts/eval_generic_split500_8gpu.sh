#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <tag> <config> <adapter_path_or_EMPTY> <out_dir> [num_tasks]" >&2
  exit 1
fi

TAG="$1"
CONFIG="$2"
ADAPTER="$3"
OUT="$4"
NUM_TASKS="${5:-8}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-3}"
GPU_LIST="${GPU_LIST:-}"

PY="${PYTHON_BIN:-/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
if [[ -n "${EVAL_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${EVAL_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"
fi

SUBSET_ROOT=/mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_500
LOG="$OUT/launcher.log"
mkdir -p "$OUT"

log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG" ; }

if [[ -n "$GPU_LIST" ]]; then
  GPU_ARRAY=(${GPU_LIST//,/ })
else
  GPU_ARRAY=()
  for rank in $(seq 0 $((NUM_TASKS - 1))); do
    GPU_ARRAY+=("$rank")
  done
fi

if [[ "${#GPU_ARRAY[@]}" -ne "$NUM_TASKS" ]]; then
  echo "GPU_LIST count (${#GPU_ARRAY[@]}) must equal num_tasks ($NUM_TASKS)" >&2
  exit 1
fi

run_refseg_split() {
  local name="$1"
  local schema="$2"
  mkdir -p "$OUT/$name"
  rm -f "$OUT/$name"/part*.json
  local pids=()
  for rank in $(seq 0 $((NUM_TASKS - 1))); do
    local gpu="${GPU_ARRAY[$rank]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.pixvl_idea1.eval.eval_refseg \
      --config "$CONFIG" \
      --adapter-path "$ADAPTER" \
      --schema-file "$schema" \
      --output "$OUT/$name/part${rank}.json" \
      --task-id "$rank" \
      --num-tasks "$NUM_TASKS" \
      > "$OUT/${name}_gpu${gpu}.log" 2>&1 &
    pids+=($!)
    sleep "$START_DELAY_SECONDS"
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  CUDA_VISIBLE_DEVICES=0 "$PY" - <<PY > "$OUT/${name}.json"
import glob, json
results = []
for path in sorted(glob.glob("$OUT/$name/part*.json")):
    results.extend(json.load(open(path)).get("results", []))
num = len(results)
mean = sum(x["ciou"] for x in results) / max(num, 1)
ap50 = sum(1 for x in results if x["ciou"] >= 0.5) / max(num, 1)
json.dump({"num_samples": num, "mean_ciou": mean, "ap50": ap50}, open("/dev/stdout", "w"), indent=2)
PY
}

run_maskcap_split() {
  local name="$1"
  local schema="$2"
  mkdir -p "$OUT/$name"
  rm -f "$OUT/$name"/part*.json
  local pids=()
  for rank in $(seq 0 $((NUM_TASKS - 1))); do
    local gpu="${GPU_ARRAY[$rank]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.pixvl_idea1.eval.eval_dlc \
      --config "$CONFIG" \
      --adapter-path "$ADAPTER" \
      --schema-file "$schema" \
      --output "$OUT/$name/part${rank}.json" \
      --task-id "$rank" \
      --num-tasks "$NUM_TASKS" \
      > "$OUT/${name}_gpu${gpu}.log" 2>&1 &
    pids+=($!)
    sleep "$START_DELAY_SECONDS"
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  CUDA_VISIBLE_DEVICES=0 "$PY" - <<PY > "$OUT/${name}.json"
import glob, json
results = []
for path in sorted(glob.glob("$OUT/$name/part*.json")):
    results.extend(json.load(open(path)).get("results", []))
num = len(results)
mean = sum(x["reward"] for x in results) / max(num, 1)
json.dump({"num_samples": num, "mean_reward": mean}, open("/dev/stdout", "w"), indent=2)
PY
}

log "start split500 eval tag=$TAG num_tasks=$NUM_TASKS"
run_maskcap_split semantic "$SUBSET_ROOT/semantic_500.jsonl"
run_refseg_split relation "$SUBSET_ROOT/relation_500.jsonl"
run_refseg_split geometry "$SUBSET_ROOT/geometry_500.jsonl"
run_refseg_split refseg_overall "$SUBSET_ROOT/refseg_val_500.jsonl"
run_maskcap_split dlc_reward "$SUBSET_ROOT/dlc_eval_100.jsonl"

CUDA_VISIBLE_DEVICES=0 "$PY" -m projects.pixvl_idea3.eval.summarize_failure_slices \
  --semantic "$OUT/semantic.json" \
  --relation "$OUT/relation.json" \
  --geometry "$OUT/geometry.json" \
  --refseg-overall "$OUT/refseg_overall.json" \
  --maskcap-overall "$OUT/dlc_reward.json" \
  --output "$OUT/summary.json" \
  > "$OUT/summary.log" 2>&1

log "split500 eval done tag=$TAG"
