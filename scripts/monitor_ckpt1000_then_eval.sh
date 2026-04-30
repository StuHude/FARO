#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/monitor_ckpt1000_then_eval.log"
UNIFIED_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_2gpu_unified_opd_rl"
ROUTED_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_rl"
ROUTED_OPD_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_opd_rl"
EVAL_SCRIPT="/mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_eval_scale100k_ckpt1000.sh"
DONE_FLAG="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_ckpt1000_eval/.eval_started"

latest_step() {
  local metrics_file="$1/metrics.partial.json"
  if [[ ! -f "$metrics_file" ]]; then
    echo "missing"
    return
  fi
  python - "$metrics_file" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
steps = data.get("steps") or []
print("none" if not steps else steps[-1].get("step", "unknown"))
PY
}

while true; do
  {
    echo "=== $(date '+%F %T %Z') ==="
    echo "unified_step=$(latest_step "$UNIFIED_DIR")"
    echo "routed_step=$(latest_step "$ROUTED_DIR")"
    echo "routed_opd_step=$(latest_step "$ROUTED_OPD_DIR")"
    [[ -d "$UNIFIED_DIR/checkpoint-step-1000" ]] && echo "unified_ckpt1000=yes" || echo "unified_ckpt1000=no"
    [[ -d "$ROUTED_DIR/checkpoint-step-1000" ]] && echo "routed_ckpt1000=yes" || echo "routed_ckpt1000=no"
    [[ -d "$ROUTED_OPD_DIR/checkpoint-step-1000" ]] && echo "routed_opd_ckpt1000=yes" || echo "routed_opd_ckpt1000=no"
  } >> "$LOG_FILE"

  if [[ -d "$UNIFIED_DIR/checkpoint-step-1000" && -d "$ROUTED_DIR/checkpoint-step-1000" && -d "$ROUTED_OPD_DIR/checkpoint-step-1000" ]]; then
    {
      echo "all_ckpt1000_ready=$(date '+%F %T %Z')"
      echo "stopping_training"
    } >> "$LOG_FILE"
    pkill -9 -f 'idea3_mvp_scale100k_2gpu_unified_opd_rl.py|idea3_mvp_scale100k_3gpu_routed_rl.py|idea3_mvp_scale100k_3gpu_routed_opd_rl.py|30520|30522|30524' || true
    sleep 5
    if [[ ! -f "$DONE_FLAG" ]]; then
      mkdir -p "$(dirname "$DONE_FLAG")"
      touch "$DONE_FLAG"
      {
        echo "starting_eval=$(date '+%F %T %Z')"
      } >> "$LOG_FILE"
      nohup bash "$EVAL_SCRIPT" > /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_ckpt1000_eval/run.log 2>&1 &
      echo $! > /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_ckpt1000_eval/run.pid
    fi
    exit 0
  fi

  sleep 1800
done
