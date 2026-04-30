#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8
LOG=$ROOT/monitor_15min.log
mkdir -p "$ROOT"
while true; do
  now=$(date '+%F %T %Z')
  echo "[$now] status check" >> "$LOG"
  python - <<'PY' >> "$LOG"
from pathlib import Path
root=Path('/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8')
checks=[
 ('refcoco_no_bucket_metric', root/'refcoco'/'no_bucket'/'metric.log'),
 ('refcoco_shuffled_metric', root/'refcoco'/'shuffled'/'metric.log'),
 ('dlc_no_bucket_eval', root/'dlc_official'/'no_bucket'/'eval.json'),
 ('dlc_shuffled_eval', root/'dlc_official'/'shuffled'/'eval.json'),
 ('split_no_bucket_relation', root/'split'/'no_bucket'/'relation.json'),
 ('split_no_bucket_geometry', root/'split'/'no_bucket'/'geometry.json'),
 ('split_no_bucket_semantic', root/'split'/'no_bucket'/'semantic.json'),
 ('split_no_bucket_overall', root/'split'/'no_bucket'/'overall.json'),
 ('split_shuffled_relation', root/'split'/'shuffled'/'relation.json'),
 ('split_shuffled_geometry', root/'split'/'shuffled'/'geometry.json'),
 ('split_shuffled_semantic', root/'split'/'shuffled'/'semantic.json'),
 ('split_shuffled_overall', root/'split'/'shuffled'/'overall.json'),
 ('summary', root/'summary_ckpt1000.json'),
 ('breakdown', root/'dlc_breakdown_ablations.json'),
]
all_done=True
for name,p in checks:
    ok=p.exists()
    print(f'{name}: {"YES" if ok else "NO"}')
    all_done = all_done and ok
print('all_done:', all_done)
PY
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$LOG" 2>/dev/null || true
  echo >> "$LOG"
  if [[ -f "$ROOT/summary_ckpt1000.json" && -f "$ROOT/dlc_breakdown_ablations.json" ]]; then
    echo "[$(date '+%F %T %Z')] monitor exit: post-eval artifacts ready" >> "$LOG"
    exit 0
  fi
  sleep 900
 done
