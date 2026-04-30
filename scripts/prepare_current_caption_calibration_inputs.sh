#!/usr/bin/env bash
set -euo pipefail

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_inputs
mkdir -p "$OUT"

"$PY" /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/prepare_caption_calibration_candidates.py \
  --official-eval-json /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/dlc_official_scores/routed_opd_rl_eval.json \
  --pred-json /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/dlc_official_preds/routed_opd_rl_pred.json \
  --qa-json /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/qa.json \
  --class-names-json /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/class_names.json \
  --output "$OUT/routed_opd_rl_hard_negatives.json" \
  --max-neg-threshold 0.65 \
  --max-recognition-error 1

"$PY" /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/prepare_caption_calibration_candidates.py \
  --official-eval-json /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/dlc_calibration_test_routed_opd_1500/eval.json \
  --pred-json /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/dlc_calibration_test_routed_opd_1500/pred.json \
  --qa-json /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/qa.json \
  --class-names-json /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/class_names.json \
  --output "$OUT/routed_opd_rl_calibrated_hard_negatives.json" \
  --max-neg-threshold 0.65 \
  --max-recognition-error 1
