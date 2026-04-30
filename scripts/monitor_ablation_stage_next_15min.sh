#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/monitor_ablation_stage_next_15min.log"
DIR_NO_BUCKET="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_opd_rl_no_bucket_opd"
DIR_SHUFFLED="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_rl_shuffled_labels"

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
    echo "no_bucket_opd_step=$(latest_step "$DIR_NO_BUCKET")"
    echo "shuffled_labels_step=$(latest_step "$DIR_SHUFFLED")"
    ps -eo pid,ppid,etime,%cpu,%mem,cmd | rg '30620|30622|no_bucket_opd|shuffled_labels|joint_routed_opd_rl_trainer' || true
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
    echo
  } >> "$LOG_FILE"
  sleep 900
done
