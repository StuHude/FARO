#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <tag> <adapter_path> <out_dir> [model_path] [num_tasks]" >&2
  exit 1
fi

TAG="$1"
ADAPTER="$2"
OUT="$3"
MODEL="${4:-/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok}"
NUM_TASKS="${5:-8}"
GPU_LIST="${GPU_LIST:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON_BIN:-/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4

VQ="${VQ_SAM2_PATH:-$MODEL/mask_tokenizer_256x2.pth}"
SAM2="${SAM2_PATH:-$MODEL/sam2.1_hiera_large.pt}"
DATASET=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench-official/DLC-bench.json
IMAGE_ROOT=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/images
QA=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dlc_bench/qa.json
CLASS_NAMES=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dlc_bench/class_names.json
JUDGE_MODEL=/mnt/pfs/xiaoyicheng/models/Meta-Llama-3.1-8B-Instruct
BASE_URL="http://127.0.0.1:9100/v1"

LOG=$OUT/launcher.log
mkdir -p "$OUT/raw_parts" "$OUT/logs"

log(){ printf '[%s] [%s] %s\n' "$(date '+%F %T')" "$TAG" "$*" | tee -a "$LOG" ; }

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

rm -f "$OUT/raw_parts"/part*.json
log "start official dlc infer ${NUM_TASKS}-way shards"
PIDS=()
for rank in $(seq 0 $((NUM_TASKS - 1))); do
  gpu="${GPU_ARRAY[$rank]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_dam_infer \
    --model_path "$MODEL" \
    --adapter_path "$ADAPTER" \
    --vq_sam2_path "$VQ" \
    --sam2_path "$SAM2" \
    --dataset "$DATASET" \
    --image_root "$IMAGE_ROOT" \
    --output_path "$OUT/raw_parts/part${rank}.json" \
    --task_id "$rank" \
    --num_tasks "$NUM_TASKS" \
    > "$OUT/logs/dlc_gpu${gpu}.log" 2>&1 &
  PIDS+=($!)
  sleep 5
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

log "merge official dlc raw shards"
CUDA_VISIBLE_DEVICES=0 "$PY" - <<PY
import glob, json
items = []
for p in sorted(glob.glob("$OUT/raw_parts/part*.json")):
    items.extend(json.load(open(p)))
json.dump(items, open("$OUT/raw.json", "w"), indent=2)
pred = {}
for item in items:
    for ms in item["mask_samples"]:
        pred[str(ms["ann_id"])] = ms["pred_caption"].replace("<|im_end|>", "").strip()
json.dump(pred, open("$OUT/pred.json", "w"), indent=2)
print(len(items), len(pred))
PY

log "start local vllm openai server"
CUDA_VISIBLE_DEVICES=0 "$PY" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 9100 \
  --model "$JUDGE_MODEL" \
  --served-model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --trust-remote-code \
  > "$OUT/logs/vllm_server.log" 2>&1 &
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

log "run official dlc vllm judge"
CUDA_VISIBLE_DEVICES=0 "$PY" /tmp/describe-anything/evaluation/eval_model_outputs.py \
  --pred "$OUT/pred.json" \
  --qa "$QA" \
  --class-names "$CLASS_NAMES" \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --base-url "$BASE_URL" \
  > "$OUT/pred_official_vllm.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PY" - <<PY
import json
src = "$OUT/pred_eval.json"
dst = "$OUT/eval.json"
data = json.load(open(src))
payload = {
    "avg_pos": data["avg_pos"],
    "avg_neg": data["avg_neg"],
    "avg": (data["avg_pos"] + data["avg_neg"]) / 2,
}
json.dump(payload, open(dst, "w"), indent=2)
print(json.dumps(payload, indent=2))
PY

log "dlc official vllm eval done"
