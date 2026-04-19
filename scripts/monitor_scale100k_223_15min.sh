#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/monitor_scale100k_223_15min.log"
UNIFIED_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_2gpu_unified_opd_rl"
ROUTED_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_rl"
ROUTED_OPD_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_opd_rl"

latest_step() {
  local metrics_file="$1/metrics.partial.json"
  if [[ ! -f "$metrics_file" ]]; then
    echo "missing"
    return
  fi
  python - "$metrics_file" <<'PY'
import json, sys
path = sys.argv[1]
try:
    data = json.load(open(path, "r", encoding="utf-8"))
    steps = data.get("steps") or []
    if not steps:
        print("none")
    else:
        print(steps[-1].get("step", "unknown"))
except Exception as exc:
    print(f"error:{exc.__class__.__name__}")
PY
}

while true; do
  {
    echo "=== $(date '+%F %T %Z') ==="
    echo "unified_step=$(latest_step "$UNIFIED_DIR")"
    echo "routed_step=$(latest_step "$ROUTED_DIR")"
    echo "routed_opd_step=$(latest_step "$ROUTED_OPD_DIR")"
    ps -eo pid,ppid,etime,%cpu,%mem,cmd | rg '30520|30522|30524|scale100k_2gpu_unified_opd_rl|scale100k_3gpu_routed_rl|scale100k_3gpu_routed_opd_rl|joint_routed_opd_rl_trainer|joint_opd_rl_trainer' || true
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
    echo
  } >> "$LOG_FILE"
  sleep 900
done
