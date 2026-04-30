#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <tag> <adapter_path> <out_dir> [model_path]" >&2
  exit 1
fi

TAG="$1"
ADAPTER="$2"
OUT="$3"
MODEL="${4:-/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok}"

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
LOG=$OUT/launcher.log
mkdir -p "$OUT/raw_parts" "$OUT/logs"

log(){ printf '[%s] [%s] %s\n' "$(date '+%F %T')" "$TAG" "$*" | tee -a "$LOG" ; }

rm -f "$OUT/raw_parts"/part*.json
log "start official dlc infer 8-way shards"
PIDS=()
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="$rank" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_dam_infer \
    --model_path "$MODEL" \
    --adapter_path "$ADAPTER" \
    --vq_sam2_path "$VQ" \
    --sam2_path "$SAM2" \
    --dataset "$DATASET" \
    --image_root "$IMAGE_ROOT" \
    --output_path "$OUT/raw_parts/part${rank}.json" \
    --task_id "$rank" \
    --num_tasks 8 \
    > "$OUT/logs/dlc_gpu${rank}.log" 2>&1 &
  PIDS+=($!)
  sleep 5
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

log "merge official dlc raw shards"
CUDA_VISIBLE_DEVICES=0 "$PY" - <<PY
import glob, json
items=[]
for p in sorted(glob.glob("$OUT/raw_parts/part*.json")):
    items.extend(json.load(open(p)))
json.dump(items, open("$OUT/raw_merged.json","w"), indent=2)
pred={}
for item in items:
    for ms in item["mask_samples"]:
        ann_id = ms.get("ann_id", ms.get("id"))
        if ann_id is None:
            continue
        pred[str(ann_id)] = ms["pred_caption"].replace("<|im_end|>", "").strip()
json.dump(pred, open("$OUT/pred.json","w"), indent=2)
print(len(items), len(pred))
PY

log "run official dlc judge with local qwen-co setting"
CUDA_VISIBLE_DEVICES=0 "$PY" /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/eval_dlc_with_local_judge.py \
  --pred "$OUT/pred.json" \
  --qa /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dlc_bench/qa.json \
  --class-names /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dlc_bench/class_names.json \
  --output "$OUT/eval.json" \
  --device cuda:0 \
  > "$OUT/logs/judge.log" 2>&1

log "dlc official eval done"
