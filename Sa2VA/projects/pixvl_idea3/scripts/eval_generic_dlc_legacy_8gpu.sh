#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <tag> <adapter_path_or_EMPTY> <out_dir> [model_path] [num_tasks]" >&2
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
if [[ -n "${EVAL_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${EVAL_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"
fi

VQ="${VQ_SAM2_PATH:-$MODEL/mask_tokenizer_256x2.pth}"
SAM2="${SAM2_PATH:-$MODEL/sam2.1_hiera_large.pt}"
DATASET=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/DLC-bench.json
IMAGE_ROOT=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/images

LOG=$OUT/launcher.log
mkdir -p "$OUT/shards"

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

rm -f "$OUT/shards"/raw_*.json
log "start legacy dlc infer ${NUM_TASKS}-way shards"
PIDS=()
for task in $(seq 0 $((NUM_TASKS - 1))); do
  gpu="${GPU_ARRAY[$task]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_dam_infer \
    --model_path "$MODEL" \
    ${ADAPTER:+--adapter_path "$ADAPTER"} \
    --vq_sam2_path "$VQ" \
    --sam2_path "$SAM2" \
    --dataset "$DATASET" \
    --image_root "$IMAGE_ROOT" \
    --task_id "$task" \
    --num_tasks "$NUM_TASKS" \
    --output_path "$OUT/shards/raw_${task}.json" \
    > "$OUT/shards/log_${task}.log" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

OUT_ENV="$OUT" "$PY" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["OUT_ENV"])
merged = []
pred = {}
for shard in sorted((root / "shards").glob("raw_*.json")):
    data = json.load(open(shard, "r", encoding="utf-8"))
    merged.extend(data)
    for item in data:
        for sample in item["mask_samples"]:
            pred[str(sample["ann_id"])] = sample["pred_caption"]
(root / "raw.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
(root / "pred.json").write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"num_raw_items": len(merged), "num_pred_items": len(pred)}, indent=2))
PY

log "legacy dlc infer done"
