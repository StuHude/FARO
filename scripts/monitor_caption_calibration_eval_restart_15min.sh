#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_eval
LOG=$ROOT/monitor_restart_15min.log
mkdir -p "$ROOT"
while true; do
  echo "[$(date '+%F %T %Z')] restart-monitor" >> "$LOG"
  ps -eo pid,etimes,cmd | rg 'qwen3vl_refcoco_padt_style_eval|qwen3vl_dam_infer|eval_refseg|eval_dlc|eval_dlc_with_local_judge' >> "$LOG" 2>/dev/null || true
  python - <<'PY' >> "$LOG"
from pathlib import Path
root=Path('/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_eval')
for name in [
 ('refcoco_metric', root/'refcoco'/'metric.log'),
 ('dlc_eval', root/'dlc_official'/'eval.json'),
 ('split_relation', root/'split'/'relation.json'),
 ('split_geometry', root/'split'/'geometry.json'),
 ('split_semantic', root/'split'/'semantic.json'),
 ('split_overall', root/'split'/'overall.json'),
 ('split_dlc_reward', root/'split'/'dlc_reward.json'),
 ('summary', root/'summary.json'),
]:
    print(name[0], 'YES' if name[1].exists() else 'NO')
PY
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$LOG" 2>/dev/null || true
  echo >> "$LOG"
  [[ -f "$ROOT/summary.json" ]] && exit 0
  sleep 900
 done
