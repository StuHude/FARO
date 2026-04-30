#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
MODEL=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok
ADAPTER=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_sft_8gpu/adapter
VQ_SAM2_PATH=$MODEL/mask_tokenizer_256x2.pth
SAM2_PATH=$MODEL/sam2.1_hiera_large.pt
DLC_JSON=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/DLC-bench.json
DLC_IMG=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/images
JUDGE=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/eval_dlc_with_local_judge.py
OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_eval/dlc_official

mkdir -p "$OUT/shards"

for task in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$task" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_dam_infer \
    --model_path "$MODEL" \
    --adapter_path "$ADAPTER" \
    --vq_sam2_path "$VQ_SAM2_PATH" \
    --sam2_path "$SAM2_PATH" \
    --dataset "$DLC_JSON" \
    --image_root "$DLC_IMG" \
    --task_id "$task" --num_tasks 8 \
    --output_path "$OUT/shards/raw_${task}.json" > "$OUT/shards/log_${task}.log" 2>&1 &
done
wait

"$PY" - <<PY
import json
from pathlib import Path
root = Path("$OUT")
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
PY

CUDA_VISIBLE_DEVICES=0 "$PY" "$JUDGE" \
  --pred "$OUT/pred.json" \
  --output "$OUT/eval.json" \
  --device cuda:0 > "$OUT/judge.log" 2>&1
