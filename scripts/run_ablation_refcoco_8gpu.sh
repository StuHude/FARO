#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export TORCHDYNAMO_DISABLE=1

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
MODEL=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok
VQ_SAM2_PATH=$MODEL/mask_tokenizer_256x2.pth
SAM2_PATH=$MODEL/sam2.1_hiera_large.pt
DATA=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json
IMG=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014
OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8/refcoco

ADAPTER_NO_BUCKET=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_opd_rl_no_bucket_opd/checkpoint-step-1000/adapter
ADAPTER_SHUFFLED=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_2gpu_routed_rl_shuffled_labels/checkpoint-step-1000/adapter

mkdir -p "$OUT/no_bucket/temp" "$OUT/no_bucket/logs" "$OUT/shuffled/temp" "$OUT/shuffled/logs"

for task in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$task" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --model_path "$MODEL" \
    --adapter_path "$ADAPTER_NO_BUCKET" \
    --vq_sam2_path "$VQ_SAM2_PATH" \
    --sam2_path "$SAM2_PATH" \
    --dataset "$DATA" \
    --image_folder "$IMG" \
    --temp_save_dir "$OUT/no_bucket/temp" \
    --task_id "$task" --num_tasks 4 > "$OUT/no_bucket/logs/shard${task}.log" 2>&1 &
done

for task in 0 1 2 3; do
  gpu=$((task+4))
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --model_path "$MODEL" \
    --adapter_path "$ADAPTER_SHUFFLED" \
    --vq_sam2_path "$VQ_SAM2_PATH" \
    --sam2_path "$SAM2_PATH" \
    --dataset "$DATA" \
    --image_folder "$IMG" \
    --temp_save_dir "$OUT/shuffled/temp" \
    --task_id "$task" --num_tasks 4 > "$OUT/shuffled/logs/shard${task}.log" 2>&1 &
done

wait

CUDA_VISIBLE_DEVICES=0 "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
  --model_path "$MODEL" \
  --adapter_path "$ADAPTER_NO_BUCKET" \
  --vq_sam2_path "$VQ_SAM2_PATH" \
  --sam2_path "$SAM2_PATH" \
  --metric-only --quiet-metric \
  --temp_save_dir "$OUT/no_bucket/temp" > "$OUT/no_bucket/metric.log" 2>&1

CUDA_VISIBLE_DEVICES=4 "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
  --model_path "$MODEL" \
  --adapter_path "$ADAPTER_SHUFFLED" \
  --vq_sam2_path "$VQ_SAM2_PATH" \
  --sam2_path "$SAM2_PATH" \
  --metric-only --quiet-metric \
  --temp_save_dir "$OUT/shuffled/temp" > "$OUT/shuffled/metric.log" 2>&1
