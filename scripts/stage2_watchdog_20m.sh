#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA"
CONFIG_PATH="/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea1/configs/idea1_joint_opd.py"
OUTPUT_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage2_joint_opd"
LOG_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/logs"
RUN_LOG="${LOG_DIR}/stage2.log"
WATCHDOG_LOG="${LOG_DIR}/stage2_watchdog_20m.log"
INTERVAL_SECONDS=1200

mkdir -p "${LOG_DIR}"

timestamp() {
  date '+%F %T'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${WATCHDOG_LOG}"
}

finished_ok() {
  /mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage2_joint_opd/metrics.json")
if not p.exists():
    raise SystemExit(1)
try:
    obj = json.loads(p.read_text())
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if obj.get("status") == "finished" else 1)
PY
}

active_launchers() {
  ps -eo pid,etimes,cmd | awk '
    /accelerate launch/ && /projects\.pixvl_idea1\.trainers\.joint_opd_trainer/ && /idea1_joint_opd\.py/ {
      print $1 " " $2
    }'
}

kill_pid_group() {
  local pid="$1"
  local pgid
  pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ' || true)"
  if [[ -n "${pgid}" ]]; then
    kill -TERM -- "-${pgid}" || true
    sleep 5
    kill -KILL -- "-${pgid}" || true
  fi
}

dedupe_launchers() {
  local launchers
  mapfile -t launchers < <(active_launchers)
  if (( ${#launchers[@]} <= 1 )); then
    return
  fi

  local newest_pid=""
  local newest_etime=""
  local line pid etime
  for line in "${launchers[@]}"; do
    pid="${line%% *}"
    etime="${line##* }"
    if [[ -z "${newest_pid}" ]] || (( etime < newest_etime )); then
      newest_pid="${pid}"
      newest_etime="${etime}"
    fi
  done

  for line in "${launchers[@]}"; do
    pid="${line%% *}"
    if [[ "${pid}" != "${newest_pid}" ]]; then
      log "dedupe_kill_old pid=${pid} keep=${newest_pid}"
      kill_pid_group "${pid}"
    fi
  done
}

while true; do
  if finished_ok; then
    log "stage2_finished"
    exit 0
  fi

  dedupe_launchers

  mapfile -t launchers < <(active_launchers)
  if (( ${#launchers[@]} == 0 )); then
    log "stage2_dead_needs_debug"
    exit 2
  else
    log "stage2_alive launchers=${#launchers[@]} latest='${launchers[0]}'"
  fi

  sleep "${INTERVAL_SECONDS}"
done
