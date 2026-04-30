#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/semcovcal_routed_opd_rl_8gpu_500_run5"
LOG="$OUT_DIR/monitor_20min.log"
while true; do
  echo "===== $(date '+%F %T %Z') =====" >> "$LOG"
  ps -eo pid,etimes,pcpu,pmem,stat,cmd | rg 'joint_routed_opd_rl_trainer|idea3_semcovcal_routed_opd_rl_8gpu_500' >> "$LOG" || true
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$LOG" || true
  python - <<'PY' >> "$LOG" 2>/dev/null
import os, json
out='/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/semcovcal_routed_opd_rl_8gpu_500_run5'
p=os.path.join(out,'metrics.partial.json')
print('metrics.partial.json', os.path.exists(p))
if os.path.exists(p):
    obj=json.load(open(p))
    print('num_steps', len(obj['steps']))
    print('last', obj['steps'][-1])
PY
  sleep 1200
done
