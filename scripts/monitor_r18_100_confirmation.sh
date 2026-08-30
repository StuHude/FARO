#!/usr/bin/env bash
set -euo pipefail

FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT="$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_seed17_100step_2gpu"
METRICS="$RUN_ROOT/metrics.json"
ADAPTER="$RUN_ROOT/adapter"
OUTPUT="$FARO_ROOT/evals/r18_100_confirmation_holdout512"
LOG="$FARO_ROOT/logs/r18_100_confirmation_monitor.log"
mkdir -p "$(dirname "$LOG")"
exec 9>"$FARO_ROOT/logs/.r18_100_confirmation.lock"
flock -n 9 || exit 0

while :; do
  if [[ -f "$METRICS" ]]; then
    state=$(python3 - "$METRICS" <<'PY'
import json, sys
try:
    d=json.load(open(sys.argv[1], encoding='utf-8'))
except Exception:
    print('waiting'); raise SystemExit
v=d.get('validity_gate') or {}
ok=(d.get('status')=='finished' and int(d.get('steps_completed',0))>=100
    and int(d.get('optimizer_updates_completed',0))>0
    and v.get('passed') is True and v.get('effective_support_gate_passed') is True
    and v.get('tail_risk_gate_passed') is True)
print('ready' if ok else d.get('status','waiting'))
PY
)
    printf '%s state=%s\n' "$(date -Is)" "$state" >> "$LOG"
    if [[ "$state" == ready ]]; then
      exec env ADAPTER="$ADAPTER" OUTPUT="$OUTPUT" \
        JOB_PREFIX=dna-fepo-r18-confirm-100step-eval POLL_SECONDS=300 \
        bash "$FARO_ROOT/scripts/submit_samtok_standalone_eval_adaptive.sh" >> "$LOG" 2>&1
    fi
  fi
  sleep 60
done
