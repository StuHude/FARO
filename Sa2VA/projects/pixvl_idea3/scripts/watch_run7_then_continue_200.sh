#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/semcovcal_routed_opd_rl_8gpu_100_run7_fast"
WATCH_LOG="$OUT_DIR/watch_then_continue.log"
CONTINUE_SCRIPT="/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/scripts/train_semcovcal_routed_opd_rl_8gpu_200_continue.sh"
CONTINUE_FLAG="$OUT_DIR/continue200_started.flag"

echo "[$(date '+%F %T %Z')] watcher started; sleeping 2.5h before first check" >> "$WATCH_LOG"
sleep 9000

while true; do
  now="$(date '+%F %T %Z')"

  if [[ -f "$CONTINUE_FLAG" ]]; then
    echo "[$now] continuation already started; exiting watcher" >> "$WATCH_LOG"
    exit 0
  fi

  finished_step="$(
    python - <<'PY' 2>/dev/null
import json
from pathlib import Path
metrics = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/semcovcal_routed_opd_rl_8gpu_100_run7_fast/metrics.json")
if metrics.exists():
    data = json.loads(metrics.read_text())
    steps = data.get("steps", [])
    if steps:
        print(int(steps[-1].get("step", -1)))
PY
  )"

  if [[ -n "${finished_step:-}" ]] && (( finished_step >= 99 )); then
    echo "[$now] detected 100-step run finished at step=$finished_step; launching continue-to-200" >> "$WATCH_LOG"
    touch "$CONTINUE_FLAG"
    nohup bash -x "$CONTINUE_SCRIPT" >> "$OUT_DIR/continue200_launcher.log" 2>&1 &
    echo "[$now] continue pid=$!" >> "$WATCH_LOG"
    exit 0
  fi

  current_step="$(
    python - <<'PY' 2>/dev/null
import json
from pathlib import Path
metrics = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/semcovcal_routed_opd_rl_8gpu_100_run7_fast/metrics.partial.json")
if metrics.exists():
    data = json.loads(metrics.read_text())
    steps = data.get("steps", [])
    if steps:
        print(int(steps[-1].get("step", -1)))
PY
  )"
  echo "[$now] 100-step run not finished yet; latest partial step=${current_step:-unknown}" >> "$WATCH_LOG"
  sleep 600
done
