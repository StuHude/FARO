#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1"
STAGE1_DIR="$OUTPUT_ROOT/stage1_joint_sft_maxmem"
STAGE2_DIR="$OUTPUT_ROOT/stage2_joint_opd"
STAGE3_DIR="$OUTPUT_ROOT/stage3_joint_opd_rl"
EVAL_DIR="$OUTPUT_ROOT/eval"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_DIR/pipeline_controller.log"

mkdir -p "$LOG_DIR"

timestamp() {
  date '+%F %T'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$LOG_FILE"
}

latest_step() {
  python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage1_joint_sft_maxmem/metrics.json"),
    Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage1_joint_sft_maxmem/metrics.partial.json"),
]
for path in paths:
    try:
        if not path.exists() or path.stat().st_size == 0:
            continue
        data = json.loads(path.read_text())
        hist = data.get("history", [])
        if hist:
            print(hist[-1].get("step", "NA"))
            raise SystemExit(0)
    except Exception:
        continue
print("NA")
PY
}

stage1_alive() {
  if pgrep -f "projects.pixvl_idea1.trainers.joint_sft_trainer --config .*idea1_joint_sft.py" >/dev/null; then
    return 0
  fi
  return 1
}

wait_for_stage1() {
  log "Waiting for current Stage 1 process to finish."
  while true; do
    if stage1_alive; then
      log "Stage 1 still running. latest_step=$(latest_step)"
      sleep 60
      continue
    fi
    break
  done
  log "Stage 1 process exited. Waiting for final adapter output."
  for _ in $(seq 1 30); do
    if [[ -f "$STAGE1_DIR/run_state.json" && -d "$STAGE1_DIR/adapter" ]]; then
      log "Stage 1 adapter is ready."
      return 0
    fi
    sleep 10
  done
  log "Stage 1 exited but final adapter was not found."
  return 1
}

run_stage() {
  local name="$1"
  local cmd="$2"
  local logfile="$3"
  log "Starting ${name}. log=${logfile}"
  stdbuf -oL -eL bash -lc "$cmd" 2>&1 | tee "$logfile"
  log "${name} finished successfully."
}

run_eval() {
  mkdir -p "$EVAL_DIR"
  log "Starting eval."
  stdbuf -oL -eL bash -lc "cd \"$REPO_ROOT/Sa2VA\" && ./projects/pixvl_idea1/scripts/run_all_eval.sh" \
    2>&1 | tee "$LOG_DIR/eval.log"
  log "Eval finished successfully."
}

main() {
  wait_for_stage1
  if [[ -f "$STAGE2_DIR/metrics.json" && -d "$STAGE2_DIR/adapter" ]]; then
    log "Stage 2 already completed. Skipping."
  else
    run_stage "Stage 2" "cd \"$REPO_ROOT/Sa2VA\" && ./projects/pixvl_idea1/scripts/train_stage2_joint_opd.sh" "$LOG_DIR/stage2.log"
  fi

  if [[ -f "$STAGE3_DIR/metrics.json" && -d "$STAGE3_DIR/adapter" ]]; then
    log "Stage 3 already completed. Skipping."
  else
    run_stage "Stage 3" "cd \"$REPO_ROOT/Sa2VA\" && ./projects/pixvl_idea1/scripts/train_stage3_joint_opd_rl.sh" "$LOG_DIR/stage3.log"
  fi

  run_eval
  log "Full pipeline completed."
}

main "$@"
