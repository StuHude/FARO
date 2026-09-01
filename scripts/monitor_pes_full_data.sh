#!/usr/bin/env bash
set -euo pipefail

FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT="$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_full_data_640step_2gpu"
SHUF_ROOT="$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_full_data_shuffled_640step_2gpu"
STATE=${STATE:-$FARO_ROOT/logs/pes_full_data_monitor}
INTERVAL=${INTERVAL:-60}
mkdir -p "$STATE" "$FARO_ROOT/logs/screen_monitor"
exec 9>"$STATE/.lock"
flock -n 9 || exit 0

ready() {
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
try:
    d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
g = d.get("validity_gate") or {}
t = d.get("tail_gppo") or {}
ok = (
    d.get("status") == "finished"
    and int(d.get("steps_completed", 0)) >= 10
    and g.get("passed") is True
    and g.get("effective_support_gate_passed") is True
    and g.get("tail_risk_gate_passed") is True
    and g.get("pes_coverage_gate_passed") is True
    and g.get("full_data_coverage_gate_passed") is True
    and int(t.get("consumed_row_count", 0)) >= 5120
)
raise SystemExit(0 if ok else 1)
PY
}

submit_eval() {
  local adapter=$1 output=$2 prefix=$3 marker=$4
  [[ -s "$marker" ]] && return 0
  if [[ -f "$output/_SUCCESS" || -f "$output/_SUCCESS_8g" || -f "$output/_SUCCESS_6g" || -f "$output/_SUCCESS_4g" || -f "$output/_SUCCESS_2g" || -f "$output/_SUCCESS_1g" ]]; then
    printf '%s\n' FINISHED > "$marker"
    return 0
  fi
  if [[ ! -f "$STATE/${prefix}.runner" ]]; then
    printf '%s\n' "$(date -Is)" > "$STATE/${prefix}.runner"
    (
      ADAPTER="$adapter" OUTPUT="$output" JOB_PREFIX="$prefix" \
        bash "$FARO_ROOT/scripts/submit_samtok_standalone_eval_adaptive.sh" \
        >> "$STATE/${prefix}.log" 2>&1 && printf '%s\n' SUBMITTED > "$marker"
    ) &
  fi
}

while :; do
  normal_metrics="$RUN_ROOT/metrics.json"
  if ready "$normal_metrics"; then
    submit_eval "$RUN_ROOT/adapter" "$FARO_ROOT/evals/predicted_evidence_scope_full_data_holdout512" \
      dna-pes-full-data-normal-eval "$STATE/normal_eval_submitted"
    if [[ ! -s "$STATE/shuffled_submit" ]]; then
      (
        JOB_NAME="dna-fepo-predicted-evidence-full-data-shuffled-640step-2g-$(date +%s)" \
          bash "$FARO_ROOT/scripts/submit_samtok_tb_gppo_predicted_evidence_scope_full_data_shuffled.sh" \
          >> "$STATE/shuffled_submit.log" 2>&1 && printf '%s\n' SUBMITTED > "$STATE/shuffled_submit"
      ) &
    fi
  fi
  if ready "$SHUF_ROOT/metrics.json"; then
    submit_eval "$SHUF_ROOT/adapter" "$FARO_ROOT/evals/predicted_evidence_scope_full_data_shuffled_holdout512" \
      dna-pes-full-data-shuffled-eval "$STATE/shuffled_eval_submitted"
  fi
  printf '%s normal=%s shuffled=%s\n' "$(date -Is)" \
    "$(ready "$normal_metrics" && echo ready || echo waiting)" \
    "$(ready "$SHUF_ROOT/metrics.json" && echo ready || echo waiting)" \
    >> "$FARO_ROOT/logs/screen_monitor/pes_full_data_monitor.log"
  sleep "$INTERVAL"
done
