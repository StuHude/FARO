#!/usr/bin/env bash

set -euo pipefail

ROOT="/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA"
OUT="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1"
LOG="$OUT/logs/stage1_watchdog.log"
STAGE1_DIR="$OUT/stage1_joint_sft_maxmem"

mkdir -p "$OUT/logs"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"
}

stage1_running() {
  pgrep -f "projects.pixvl_idea1.trainers.joint_sft_trainer --config .*idea1_joint_sft.py" >/dev/null
}

stage2_running() {
  pgrep -f "projects.pixvl_idea1.trainers.joint_opd_trainer --config .*idea1_joint_opd.py" >/dev/null
}

stage1_complete() {
  [[ -f "$STAGE1_DIR/metrics.json" && -f "$STAGE1_DIR/run_state.json" && -f "$STAGE1_DIR/adapter/adapter_model.safetensors" ]]
}

latest_step() {
  /mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python - <<'PY' 2>/dev/null || true
import json
from pathlib import Path
p = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage1_joint_sft_maxmem/metrics.partial.json")
if not p.exists() or p.stat().st_size == 0:
    print("NA")
else:
    data = json.loads(p.read_text())
    hist = data.get("steps", []) or data.get("history", [])
    print(hist[-1].get("step", "NA") if hist else "NA")
PY
}

start_stage1() {
  log "stage1 missing; restart"
  cd "$ROOT"
  setsid ./projects/pixvl_idea1/scripts/train_stage1_joint_sft.sh >> "$OUT/logs/stage1_detached.log" 2>&1 < /dev/null &
}

start_chain() {
  log "pipeline auto-start disabled"
}

log "watchdog start"
while true; do
  if stage1_running; then
    log "stage1 running step=$(latest_step)"
  else
    if stage1_complete; then
      log "stage1 complete"
      start_chain
      if stage2_running; then
        log "stage2 running"
      fi
    else
      start_stage1
    fi
  fi
  start_chain
  sleep 1200
done
