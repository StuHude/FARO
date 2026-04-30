#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export TORCHDYNAMO_DISABLE=1

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
MODEL=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok
ADAPTER=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_sft_8gpu/adapter
VQ_SAM2_PATH=$MODEL/mask_tokenizer_256x2.pth
SAM2_PATH=$MODEL/sam2.1_hiera_large.pt
DATA=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json
IMG=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014
OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_eval/refcoco

mkdir -p "$OUT/logs" "$OUT/temp"

for task in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$task" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --model_path "$MODEL" \
    --adapter_path "$ADAPTER" \
    --vq_sam2_path "$VQ_SAM2_PATH" \
    --sam2_path "$SAM2_PATH" \
    --dataset "$DATA" \
    --image_folder "$IMG" \
    --temp_save_dir "$OUT/temp" \
    --task_id "$task" --num_tasks 8 > "$OUT/logs/shard${task}.log" 2>&1 &
done
wait

CUDA_VISIBLE_DEVICES=0 "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
  --model_path "$MODEL" \
  --adapter_path "$ADAPTER" \
  --vq_sam2_path "$VQ_SAM2_PATH" \
  --sam2_path "$SAM2_PATH" \
  --metric-only --quiet-metric \
  --temp_save_dir "$OUT/temp" > "$OUT/metric.log" 2>&1
