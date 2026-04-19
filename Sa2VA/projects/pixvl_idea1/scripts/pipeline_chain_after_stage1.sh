#!/usr/bin/env bash

set -euo pipefail

ROOT="/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA"
OUT="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1"
LOG="$OUT/logs/pipeline_chain.log"
STAGE1="$OUT/stage1_joint_sft_maxmem"

mkdir -p "$OUT/logs"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"
}

log "wait stage1"
while true; do
  if pgrep -f "projects.pixvl_idea1.trainers.joint_sft_trainer --config .*idea1_joint_sft.py" >/dev/null; then
    sleep 60
    continue
  fi
  if [[ -f "$STAGE1/metrics.json" && -f "$STAGE1/run_state.json" && -f "$STAGE1/adapter/adapter_model.safetensors" ]]; then
    break
  fi
  log "stage1 incomplete; continue waiting"
  sleep 60
done

cd "$ROOT"

log "start stage2"
./projects/pixvl_idea1/scripts/train_stage2_joint_opd.sh >> "$OUT/logs/stage2.log" 2>&1

log "start stage3"
./projects/pixvl_idea1/scripts/train_stage3_joint_opd_rl.sh >> "$OUT/logs/stage3.log" 2>&1

log "start eval"
./projects/pixvl_idea1/scripts/run_all_eval.sh >> "$OUT/logs/eval.log" 2>&1

log "done"
