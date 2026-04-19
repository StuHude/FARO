#!/usr/bin/env bash

set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
BASE_MODEL=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co
VQ=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/mask_tokenizer_256x2.pth
SAM2=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/sam2.1_hiera_large.pt
IMG=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014
PADT_ROOT=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO
DATA=${PADT_ROOT}/refcoco_val.json
LOG_ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/mvp_official_refcoco

mkdir -p "$LOG_ROOT"

"$PY" -m projects.samtok.evaluation.qwen3vl.export_refcoco_padt_val --source refcoco
TARGET=$(wc -l < "$DATA")
echo "TARGET_REFCOCO_VAL=$TARGET"

eval_model() {
  local name="$1"
  local gpu_a="$2"
  local gpu_b="$3"
  local adapter="${4:-}"
  local temp_dir="/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/temp_save/refcoco_${name}"
  local log_prefix="${LOG_ROOT}/${name}"

  rm -rf "$temp_dir"
  mkdir -p "$temp_dir"

  echo "=== START ${name} $(date '+%F %T') ==="
  if [ -n "$adapter" ]; then
    CUDA_VISIBLE_DEVICES="$gpu_a" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path "$BASE_MODEL" \
      --adapter_path "$adapter" \
      --vq_sam2_path "$VQ" \
      --sam2_path "$SAM2" \
      --dataset "$DATA" \
      --image_folder "$IMG" \
      --temp_save_dir "$temp_dir" \
      --task_id 0 \
      --num_tasks 2 >> "${log_prefix}_shard0.log" 2>&1 &
    local pid0=$!
    CUDA_VISIBLE_DEVICES="$gpu_b" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path "$BASE_MODEL" \
      --adapter_path "$adapter" \
      --vq_sam2_path "$VQ" \
      --sam2_path "$SAM2" \
      --dataset "$DATA" \
      --image_folder "$IMG" \
      --temp_save_dir "$temp_dir" \
      --task_id 1 \
      --num_tasks 2 >> "${log_prefix}_shard1.log" 2>&1 &
    local pid1=$!
  else
    CUDA_VISIBLE_DEVICES="$gpu_a" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path "$BASE_MODEL" \
      --vq_sam2_path "$VQ" \
      --sam2_path "$SAM2" \
      --dataset "$DATA" \
      --image_folder "$IMG" \
      --temp_save_dir "$temp_dir" \
      --task_id 0 \
      --num_tasks 2 >> "${log_prefix}_shard0.log" 2>&1 &
    local pid0=$!
    CUDA_VISIBLE_DEVICES="$gpu_b" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path "$BASE_MODEL" \
      --vq_sam2_path "$VQ" \
      --sam2_path "$SAM2" \
      --dataset "$DATA" \
      --image_folder "$IMG" \
      --temp_save_dir "$temp_dir" \
      --task_id 1 \
      --num_tasks 2 >> "${log_prefix}_shard1.log" 2>&1 &
    local pid1=$!
  fi

  while true; do
    local count
    count=$(find "$temp_dir" -maxdepth 1 -type f | wc -l)
    echo "[${name}] $(date '+%F %T') ${count}/${TARGET}"
    if [ "$count" -ge "$TARGET" ]; then
      break
    fi
    if ! kill -0 "$pid0" 2>/dev/null && ! kill -0 "$pid1" 2>/dev/null; then
      break
    fi
    sleep 120
  done

  wait "$pid0"
  wait "$pid1"

  echo "=== METRIC ${name} $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="$gpu_a" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --metric-only \
    --temp_save_dir "$temp_dir" > "${log_prefix}_metric.log" 2>&1
}

eval_model base 0 1 "" &
pid_base=$!
eval_model unified 2 3 /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/quick_unified_opd_rl_single/adapter &
pid_unified=$!
eval_model routed_rl 4 5 /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/quick_routed_rl_single/adapter &
pid_routed_rl=$!
eval_model routed_opd_rl 6 7 /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/quick_routed_opd_rl_single/adapter &
pid_routed_opd=$!

wait "$pid_base"
wait "$pid_unified"
wait "$pid_routed_rl"
wait "$pid_routed_opd"

echo "=== DONE $(date '+%F %T') ==="
