#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
MODEL=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok
VQ_SAM2_PATH=$MODEL/mask_tokenizer_256x2.pth
SAM2_PATH=$MODEL/sam2.1_hiera_large.pt
DLC_JSON=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/DLC-bench.json
DLC_IMG=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/images
JUDGE=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/eval_dlc_with_local_judge.py
OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8/dlc_official

ADAPTER_NO_BUCKET=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_opd_rl_no_bucket_opd/checkpoint-step-1000/adapter
ADAPTER_SHUFFLED=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_2gpu_routed_rl_shuffled_labels/checkpoint-step-1000/adapter

mkdir -p "$OUT/no_bucket/shards" "$OUT/shuffled/shards"

for task in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$task" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_dam_infer \
    --model_path "$MODEL" \
    --adapter_path "$ADAPTER_NO_BUCKET" \
    --vq_sam2_path "$VQ_SAM2_PATH" \
    --sam2_path "$SAM2_PATH" \
    --dataset "$DLC_JSON" \
    --image_root "$DLC_IMG" \
    --task_id "$task" --num_tasks 4 \
    --output_path "$OUT/no_bucket/shards/raw_${task}.json" > "$OUT/no_bucket/shards/log_${task}.log" 2>&1 &
done

for task in 0 1 2 3; do
  gpu=$((task+4))
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_dam_infer \
    --model_path "$MODEL" \
    --adapter_path "$ADAPTER_SHUFFLED" \
    --vq_sam2_path "$VQ_SAM2_PATH" \
    --sam2_path "$SAM2_PATH" \
    --dataset "$DLC_JSON" \
    --image_root "$DLC_IMG" \
    --task_id "$task" --num_tasks 4 \
    --output_path "$OUT/shuffled/shards/raw_${task}.json" > "$OUT/shuffled/shards/log_${task}.log" 2>&1 &
done

wait

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8/dlc_official")
for name in ["no_bucket", "shuffled"]:
    merged = []
    pred = {}
    for shard in sorted((root / name / "shards").glob("raw_*.json")):
        data = json.load(open(shard, "r", encoding="utf-8"))
        merged.extend(data)
        for item in data:
            for sample in item["mask_samples"]:
                pred[str(sample["ann_id"])] = sample["pred_caption"]
    (root / name / "raw.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / name / "pred.json").write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
PY

CUDA_VISIBLE_DEVICES=0 "$PY" "$JUDGE" \
  --pred "$OUT/no_bucket/pred.json" \
  --output "$OUT/no_bucket/eval.json" \
  --device cuda:0 > "$OUT/no_bucket/judge.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=4 "$PY" "$JUDGE" \
  --pred "$OUT/shuffled/pred.json" \
  --output "$OUT/shuffled/eval.json" \
  --device cuda:0 > "$OUT/shuffled/judge.log" 2>&1 &
P1=$!
wait "$P0" "$P1"
