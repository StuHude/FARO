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

GREF_SCHEMA=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/grefcoco_val.jsonl
PARTS="$OUT/gref_parts"
LOG="$OUT/launcher.log"

mkdir -p "$OUT" "$PARTS"
rm -f "$PARTS"/grefcoco.part*.json

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
log "start grefcoco eval tag=$TAG num_tasks=$NUM_TASKS"
PIDS=()
for rank in $(seq 0 $((NUM_TASKS - 1))); do
  gpu="${GPU_ARRAY[$rank]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.pixvl_idea1.eval.eval_refseg \
    --config "$CONFIG" \
    --adapter-path "$ADAPTER" \
    --schema-file "$GREF_SCHEMA" \
    --output "$PARTS/grefcoco.part${rank}.json" \
    --task-id "$rank" \
    --num-tasks "$NUM_TASKS" \
    > "$OUT/grefcoco_gpu${gpu}.log" 2>&1 &
  PIDS+=($!)
  sleep "$START_DELAY_SECONDS"
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

log "merge grefcoco parts"
CUDA_VISIBLE_DEVICES=0 "$PY" - <<PY > "$OUT/grefcoco_metrics.json"
import glob, json

results = []
for path in sorted(glob.glob("$PARTS/grefcoco.part*.json")):
    results.extend(json.load(open(path)).get("results", []))
num = len(results)
mean_ciou = sum(x["ciou"] for x in results) / max(num, 1)
ap50 = sum(1 for x in results if x["ciou"] >= 0.5) / max(num, 1)
json.dump({"num_samples": num, "mean_ciou": mean_ciou, "ap50": ap50}, open("/dev/stdout", "w"), indent=2)
PY

log "grefcoco eval done tag=$TAG"
