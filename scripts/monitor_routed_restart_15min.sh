#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/monitor_routed_restart_15min.log"
RL_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/2gpu_globalbs4_2000_routed_rl"
OPD_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/2gpu_globalbs4_2000_routed_opd_rl"

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
    echo "routed_rl_step=$(latest_step "$RL_DIR")"
    echo "routed_opd_rl_step=$(latest_step "$OPD_DIR")"
    ps -eo pid,ppid,etime,%cpu,%mem,cmd | rg 'idea3_mvp_2gpu_globalbs4_2000_routed_rl.py|idea3_mvp_2gpu_globalbs4_2000_routed_opd_rl.py|joint_routed_opd_rl_trainer' || true
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader || true
    echo
  } >> "$LOG_FILE"
  sleep 900
done
