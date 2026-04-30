#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
MODEL=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok
ADAPTER=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_opd_rl/checkpoint-step-1500/adapter
VQ_SAM2_PATH=$MODEL/mask_tokenizer_256x2.pth
SAM2_PATH=$MODEL/sam2.1_hiera_large.pt
DLC_JSON=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/DLC-bench.json
DLC_IMG=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/images
JUDGE_SCRIPT=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/eval_dlc_with_local_judge.py
OUT_DIR=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/dlc_calibration_test_routed_opd_1500

mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_dam_infer \
  --model_path "$MODEL" \
  --adapter_path "$ADAPTER" \
  --vq_sam2_path "$VQ_SAM2_PATH" \
  --sam2_path "$SAM2_PATH" \
  --dataset "$DLC_JSON" \
  --image_root "$DLC_IMG" \
  --max_new_tokens 128 \
  --prompt_suffix "Only mention visually certain details. If unsure, omit the detail rather than guessing." \
  --output_path "$OUT_DIR/raw.json" > "$OUT_DIR/generate.log" 2>&1

"$PY" - <<'PY'
import json
from pathlib import Path

raw = json.load(open("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/dlc_calibration_test_routed_opd_1500/raw.json", "r", encoding="utf-8"))
pred = {}
for item in raw:
    for sample in item["mask_samples"]:
        pred[str(sample["ann_id"])] = sample["pred_caption"]
Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/dlc_calibration_test_routed_opd_1500/pred.json").write_text(
    json.dumps(pred, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" "$JUDGE_SCRIPT" \
  --pred "$OUT_DIR/pred.json" \
  --output "$OUT_DIR/eval.json" \
  --device cuda:0 > "$OUT_DIR/judge.log" 2>&1
