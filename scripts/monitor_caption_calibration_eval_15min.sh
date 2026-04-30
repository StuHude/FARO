#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_eval
LOG=$ROOT/monitor_15min.log
mkdir -p "$ROOT"
while true; do
  echo "[$(date '+%F %T %Z')] status" >> "$LOG"
  python - <<'PY' >> "$LOG"
from pathlib import Path
root=Path('/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_eval')
checks=[
 ('refcoco_metric', root/'refcoco'/'metric.log'),
 ('dlc_eval', root/'dlc_official'/'eval.json'),
 ('split_relation', root/'split'/'relation.json'),
 ('split_geometry', root/'split'/'geometry.json'),
 ('split_semantic', root/'split'/'semantic.json'),
 ('split_overall', root/'split'/'overall.json'),
 ('split_dlc_reward', root/'split'/'dlc_reward.json'),
 ('summary', root/'summary.json'),
]
for name,p in checks:
    print(name, 'YES' if p.exists() else 'NO')
PY
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$LOG" 2>/dev/null || true
  echo >> "$LOG"
  [[ -f "$ROOT/summary.json" ]] && exit 0
  sleep 900
 done
