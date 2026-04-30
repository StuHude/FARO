#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_sft_8gpu
LOG=$ROOT/monitor_15min.log
mkdir -p "$ROOT"
while true; do
  echo "[$(date '+%F %T %Z')] recognition_negsup_sft check" >> "$LOG"
  ps -eo pid,etimes,%cpu,%mem,cmd | rg 'idea3_recognition_negsup_sft_8gpu|joint_sft_trainer --config .*/idea3_recognition_negsup_sft_8gpu.py|accelerate launch --main_process_port .*idea3_recognition_negsup_sft_8gpu.py' >> "$LOG" 2>/dev/null || true
  python - <<'PY' >> "$LOG"
import json
from pathlib import Path
p=Path('/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_sft_8gpu/metrics.partial.json')
print('metrics_partial', p.exists())
if p.exists():
 d=json.load(open(p))
 print('last_step', d['steps'][-1]['step'])
pm=Path('/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_sft_8gpu/metrics.json')
print('metrics_final', pm.exists())
PY
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$LOG" 2>/dev/null || true
  echo >> "$LOG"
  [[ -f "$ROOT/metrics.json" ]] && exit 0
  sleep 900
 done
