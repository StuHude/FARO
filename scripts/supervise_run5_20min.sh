#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/semcovcal_routed_opd_rl_8gpu_500_run5"
LOG="$OUT_DIR/supervise_20min.log"
TRAIN_SCRIPT="/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/scripts/train_semcovcal_routed_opd_rl_8gpu_500.sh"
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
  if ! ps -eo cmd | rg -q 'joint_routed_opd_rl_trainer.*idea3_semcovcal_routed_opd_rl_8gpu_500.py'; then
    echo 'training_missing -> restart' >> "$LOG"
    for f in "$OUT_DIR"/manual_logs/rank*.log; do
      [ -f "$f" ] && { echo "--- $(basename "$f") ---" >> "$LOG"; tail -n 40 "$f" >> "$LOG"; }
    done
    nohup bash "$TRAIN_SCRIPT" >> "$OUT_DIR/train_top.log" 2>&1 &
    echo "restarted_pid=$!" >> "$LOG"
  fi
  sleep 1200
done
