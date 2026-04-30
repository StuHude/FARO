#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/semcovcal_routed_opd_rl_8gpu_500"
LOG="$OUT_DIR/monitor_20min_until_60g_30min.log"
TARGET_MB=$((60*1024))
START_OK_TS=""
while true; do
  now=$(date '+%F %T %Z')
  echo "===== $now =====" >> "$LOG"
  ps -eo pid,etimes,pcpu,pmem,stat,cmd | rg 'joint_routed_opd_rl_trainer|idea3_semcovcal_routed_opd_rl_8gpu_500' >> "$LOG" || true
  mapfile -t lines < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '%s\n' "${lines[@]}" >> "$LOG"
  p="$OUT_DIR/metrics.partial.json"
  if [ -f "$p" ]; then
    python - <<'PY' >> "$LOG" 2>/dev/null
import json
p='/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/semcovcal_routed_opd_rl_8gpu_500/metrics.partial.json'
obj=json.load(open(p))
print('num_steps', len(obj['steps']))
print('last', obj['steps'][-1])
PY
  else
    echo 'metrics.partial.json missing' >> "$LOG"
  fi
  all_ok=1
  for line in "${lines[@]}"; do
    mem=$(echo "$line" | awk -F',' '{gsub(/ /, "", $2); print $2}')
    if [ -z "$mem" ] || [ "$mem" -lt "$TARGET_MB" ]; then
      all_ok=0
      break
    fi
  done
  if [ "$all_ok" -eq 1 ]; then
    if [ -z "$START_OK_TS" ]; then
      START_OK_TS=$(date +%s)
      echo "all_gpus_over_60g_since=$START_OK_TS" >> "$LOG"
    fi
    now_ts=$(date +%s)
    dur=$((now_ts-START_OK_TS))
    echo "all_gpus_over_60g_duration_sec=$dur" >> "$LOG"
    if [ "$dur" -ge 1800 ]; then
      echo "SUCCESS_CONDITION_MET at $now" >> "$LOG"
      break
    fi
  else
    START_OK_TS=""
  fi
  sleep 1200
done
