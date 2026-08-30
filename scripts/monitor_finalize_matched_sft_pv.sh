#!/usr/bin/env bash
set -euo pipefail

FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
STATE=${STATE:-$FARO_ROOT/logs/screen_monitor/finalize}
mkdir -p "$STATE"
exec 9>"$STATE/.lock"
flock -n 9 || exit 0

PV_R18="$FARO_ROOT/evals/paired_view_vs_r18_bootstrap20k.json"
PV_SFT="$FARO_ROOT/evals/paired_view_vs_matched_sft_bootstrap20k.json"
PV_DECISION="$FARO_ROOT/evals/pv_training_gate.json"
SFT_R18="$FARO_ROOT/evals/r18_matched_sft_vs_r18_bootstrap20k.json"

pv_decision_closed() {
  [[ -s "$PV_DECISION" ]] || return 1
  python3 - "$PV_DECISION" <<'PY'
import json, sys
from pathlib import Path
try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if value.get("decision") == "closed_training_gate" else 1)
PY
}

pv_ready() {
  if [[ -s "$PV_R18" && -s "$PV_SFT" ]]; then
    return 0
  fi
  pv_decision_closed
}

# The finalizer is intentionally idempotent: the analysis helper also exits
# zero when the PV file is absent, so completion must be determined from both
# required artifacts rather than from its return code alone.
while [[ ! -s "$SFT_R18" || ( ! -s "$PV_R18" && ! -s "$PV_DECISION" ) ]]; do
  set +e
  FARO_ROOT="$FARO_ROOT" bash "$FARO_ROOT/scripts/finalize_matched_sft_pv.sh" \
    >> "$STATE/monitor.log" 2>&1
  rc=$?
  set -e
  # The helper can return zero after processing the matched-SFT control even
  # when PV is still absent. Only the artifact checks above define completion.
  if [[ ! -s "$SFT_R18" || ! pv_ready ]]; then
    printf '%s waiting_for_pv rc=%s\n' "$(date -Is)" "$rc" >> "$STATE/monitor.log"
    sleep 300
  else
    break
  fi
done

# Continue with the next isolated hypothesis only after both paired-view
# comparisons are complete and the candidate is closed by its own gate.
if [[ -s "$SFT_R18" && ( -s "$PV_R18" || -s "$PV_DECISION" ) ]]; then
  STATE="$STATE/r35_submit" bash "$FARO_ROOT/scripts/submit_r35_after_pv_decision.sh" \
    >> "$STATE/r35_submit.log" 2>&1 || true
fi
