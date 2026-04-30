#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_sft_8gpu_longrun
LOG=$ROOT/monitor_20min_until_1h.log
mkdir -p "$ROOT"
while true; do
  echo "[$(date '+%F %T %Z')] longrun 20min check" >> "$LOG"
  ps -eo pid,etimes,%cpu,%mem,cmd | rg 'idea3_recognition_negsup_sft_8gpu_longrun|joint_sft_trainer --config .*/idea3_recognition_negsup_sft_8gpu_longrun.py|accelerate launch --main_process_port .*idea3_recognition_negsup_sft_8gpu_longrun.py' >> "$LOG" 2>/dev/null || true
  python - <<'PY' >> "$LOG"
import json
from pathlib import Path
p=Path('/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_sft_8gpu_longrun/metrics.partial.json')
print('metrics_partial', p.exists())
if p.exists():
 d=json.load(open(p))
 print('steps_recorded', len(d['steps']))
 print('last_step', d['steps'][-1]['step'])
pm=Path('/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_sft_8gpu_longrun/metrics.json')
print('metrics_final', pm.exists())
PY
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$LOG" 2>/dev/null || true
  echo >> "$LOG"
  python - <<'PY'
import subprocess, re, sys
cmd="ps -eo etimes,cmd | rg 'accelerate launch --main_process_port .*idea3_recognition_negsup_sft_8gpu_longrun.py|joint_sft_trainer --config .*/idea3_recognition_negsup_sft_8gpu_longrun.py'"
proc=subprocess.run(cmd, shell=True, capture_output=True, text=True)
text=proc.stdout.strip().splitlines()
max_elapsed=0
for line in text:
    m=re.match(r'\s*(\d+)', line)
    if m:
        max_elapsed=max(max_elapsed, int(m.group(1)))
if max_elapsed >= 3600:
    sys.exit(10)
PY
  status=$?
  if [[ $status -eq 10 ]]; then
    echo "[$(date '+%F %T %Z')] monitor exit: run exceeded 1h" >> "$LOG"
    exit 0
  fi
  if [[ -f "$ROOT/metrics.json" ]]; then
    echo "[$(date '+%F %T %Z')] monitor exit: training finished before 1h threshold" >> "$LOG"
    exit 0
  fi
  sleep 1200
 done
