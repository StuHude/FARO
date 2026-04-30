#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export PIXVL_TEXT_SIM_DEVICE=cpu
export PIXVL_TEXT_SIM_LOCAL_ONLY=1
export TORCHDYNAMO_DISABLE=1

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
ADAPTER=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_sft_8gpu/adapter
CONFIG=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_caption_calibration_sft_8gpu.py
SUBSET_ROOT=/mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_2000
OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_eval/split

mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 "$PY" -m projects.pixvl_idea1.eval.eval_refseg --config "$CONFIG" --adapter-path "$ADAPTER" --schema-file "$SUBSET_ROOT/relation_2000.jsonl" --output "$OUT/relation.json" &
CUDA_VISIBLE_DEVICES=1 "$PY" -m projects.pixvl_idea1.eval.eval_refseg --config "$CONFIG" --adapter-path "$ADAPTER" --schema-file "$SUBSET_ROOT/geometry_2000.jsonl" --output "$OUT/geometry.json" &
CUDA_VISIBLE_DEVICES=2 "$PY" -m projects.pixvl_idea1.eval.eval_dlc --config "$CONFIG" --adapter-path "$ADAPTER" --schema-file "$SUBSET_ROOT/semantic_2000.jsonl" --output "$OUT/semantic.json" &
CUDA_VISIBLE_DEVICES=3 "$PY" -m projects.pixvl_idea1.eval.eval_refseg --config "$CONFIG" --adapter-path "$ADAPTER" --schema-file "$SUBSET_ROOT/refseg_val_2000.jsonl" --output "$OUT/overall.json" &
CUDA_VISIBLE_DEVICES=4 "$PY" -m projects.pixvl_idea1.eval.eval_dlc --config "$CONFIG" --adapter-path "$ADAPTER" --schema-file "$SUBSET_ROOT/dlc_eval_100.jsonl" --output "$OUT/dlc_reward.json" &
wait
