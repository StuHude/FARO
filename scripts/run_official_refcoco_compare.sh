#!/usr/bin/env bash

set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
BASE=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co
ADAPTER=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage1_joint_sft_maxmem/checkpoint-step-800/adapter
VQ=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/mask_tokenizer_256x2.pth
SAM2=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/sam2.1_hiera_large.pt
IMG=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014
DATA=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json
LOGROOT=/mnt/pfs/xiaoyicheng/outputs/samtok_refcoco_full_official
mkdir -p "$LOGROOT"

pkill -f "qwen3vl_refcoco_padt_style_eval --model_path /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co" || true
sleep 5

"$PY" -m projects.samtok.evaluation.qwen3vl.export_refcoco_padt_val --source refcoco
TARGET=$(wc -l < "$DATA")

eval_one() {
  local name="$1"
  local adapter="$2"
  local temp="/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/temp_save/refcoco_${name}"
  rm -rf "$temp"
  mkdir -p "$temp"
  echo "=== START $name $(date '+%F %T') ==="
  local pids=()
  for gpu in 0 1 2 3 4 5 6 7; do
    if [ -n "$adapter" ]; then
      CUDA_VISIBLE_DEVICES=$gpu "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
        --model_path "$BASE" \
        --adapter_path "$adapter" \
        --vq_sam2_path "$VQ" \
        --sam2_path "$SAM2" \
        --dataset "$DATA" \
        --image_folder "$IMG" \
        --temp_save_dir "$temp" \
        --task_id "$gpu" \
        --num_tasks 8 >> "$LOGROOT/${name}_shard${gpu}.log" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES=$gpu "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
        --model_path "$BASE" \
        --vq_sam2_path "$VQ" \
        --sam2_path "$SAM2" \
        --dataset "$DATA" \
        --image_folder "$IMG" \
        --temp_save_dir "$temp" \
        --task_id "$gpu" \
        --num_tasks 8 >> "$LOGROOT/${name}_shard${gpu}.log" 2>&1 &
    fi
    pids+=("$!")
  done
  while true; do
    local count
    count=$(find "$temp" -maxdepth 1 -type f | wc -l)
    echo "[$name] $(date '+%F %T') ${count}/${TARGET}"
    if [ "$count" -ge "$TARGET" ]; then
      break
    fi
    sleep 300
  done
  wait "${pids[@]}"
  echo "=== METRIC $name $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES=0 "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --metric-only \
    --temp_save_dir "$temp"
}

eval_one base ""
eval_one stage1 "$ADAPTER"
