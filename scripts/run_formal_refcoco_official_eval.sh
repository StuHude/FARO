#!/usr/bin/env bash

set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
BASE=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co
VQ=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/mask_tokenizer_256x2.pth
SAM2=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/sam2.1_hiera_large.pt
IMG=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014
DATA=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json
TARGET=$(wc -l < "$DATA")
LOG_ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/formal_official_refcoco

mkdir -p "$LOG_ROOT"

run_eval() {
  local name="$1"
  local gpu0="$2"
  local gpu1="$3"
  local adapter="$4"
  local temp_dir="/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/temp_save/refcoco_${name}_formal"
  local log_dir="${LOG_ROOT}/${name}"
  mkdir -p "$log_dir"
  rm -rf "$temp_dir"
  mkdir -p "$temp_dir"

  CUDA_VISIBLE_DEVICES=$gpu0 "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --model_path "$BASE" \
    --adapter_path "$adapter" \
    --vq_sam2_path "$VQ" \
    --sam2_path "$SAM2" \
    --dataset "$DATA" \
    --image_folder "$IMG" \
    --temp_save_dir "$temp_dir" \
    --task_id 0 \
    --num_tasks 2 >> "${log_dir}/shard0.log" 2>&1 &
  local pid0=$!

  CUDA_VISIBLE_DEVICES=$gpu1 "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --model_path "$BASE" \
    --adapter_path "$adapter" \
    --vq_sam2_path "$VQ" \
    --sam2_path "$SAM2" \
    --dataset "$DATA" \
    --image_folder "$IMG" \
    --temp_save_dir "$temp_dir" \
    --task_id 1 \
    --num_tasks 2 >> "${log_dir}/shard1.log" 2>&1 &
  local pid1=$!

  while true; do
    local count
    count=$(find "$temp_dir" -maxdepth 1 -type f | wc -l)
    printf '[%s] %s %s/%s\n' "$name" "$(date '+%F %T')" "$count" "$TARGET"
    if [ "$count" -ge "$TARGET" ]; then
      break
    fi
    sleep 180
  done

  wait "$pid0"
  wait "$pid1"

  CUDA_VISIBLE_DEVICES=$gpu0 "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --metric-only \
    --quiet-metric \
    --temp_save_dir "$temp_dir" > "${log_dir}/metric.log" 2>&1
}

run_eval formal_unified 0 1 /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/formal_unified_opd_rl_single/adapter &
PID_UNIFIED=$!
run_eval formal_routed_rl 2 3 /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/formal_routed_rl_single/adapter &
PID_ROUTED_RL=$!
run_eval formal_routed_opd 4 5 /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/formal_routed_opd_rl_single/adapter &
PID_ROUTED_OPD=$!

wait "$PID_UNIFIED"
wait "$PID_ROUTED_RL"
wait "$PID_ROUTED_OPD"
