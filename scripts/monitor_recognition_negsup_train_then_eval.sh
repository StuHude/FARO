#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_sft_8gpu
EVAL_ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_eval
LOG=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/monitor_recognition_negsup_train_then_eval.log

mkdir -p /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3

while true; do
  echo "[$(date '+%F %T %Z')] train-then-eval monitor" >> "$LOG"
  if [[ -f "$TRAIN_ROOT/metrics.json" ]]; then
    echo "[$(date '+%F %T %Z')] training finished, launch eval" >> "$LOG"
    break
  fi
  sleep 1200
done

mkdir -p "$EVAL_ROOT"
bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/launch_recognition_negsup_eval_all3.sh >> "$LOG" 2>&1

while true; do
  if [[ -f "$EVAL_ROOT/summary.json" ]]; then
    echo "[$(date '+%F %T %Z')] eval finished" >> "$LOG"
    exit 0
  fi
  sleep 1200
done
