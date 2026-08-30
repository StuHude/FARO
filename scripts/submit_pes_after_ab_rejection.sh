#!/usr/bin/env bash
set -euo pipefail
FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
AB_EVAL=${AB_EVAL:-$FARO_ROOT/evals/action_budget_native_rank_local_vs_matched_sft_bootstrap20k.json}
STATE=${STATE:-$FARO_ROOT/logs/pes_submit}
INTERVAL=${INTERVAL:-300}
mkdir -p "$STATE"
exec 9>"$STATE/.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$STATE/pid"
cleanup_pid() {
  [[ "$(sed -n '1p' "$STATE/pid" 2>/dev/null || true)" == "$$" ]] && rm -f "$STATE/pid"
}
trap cleanup_pid EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ab_rejected() {
  python3 - "$AB_EVAL" <<'PY'
import json, sys
from pathlib import Path
try:
    d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
ok = (d.get("promotion_gate") is False or d.get("ci_corrected_promotion_gate") is False)
raise SystemExit(0 if ok else 1)
PY
}
marker="$STATE/submitted"
while [[ ! -s "$marker" ]]; do
  ab_rejected || { sleep "$INTERVAL"; continue; }
  set +e
  listing=$(rjob list --namespace=ailab-dnacoding 2>&1)
  rc=$?
  set -e
  if (( rc != 0 )); then
    printf '%s control_plane_unavailable status=%s\n' "$(date -Is)" "$rc" >> "$STATE/submit.log"
    sleep "$INTERVAL"; continue
  fi
  existing=$(printf '%s\n' "$listing" | grep -E 'dna-fepo-predicted-evidence-scope-10step-2g' | head -1 || true)
  if [[ -n "$existing" ]]; then
    printf '%s\n' "$(printf '%s\n' "$existing" | awk '{print $1}')" > "$marker"; break
  fi
  job_name="dna-fepo-predicted-evidence-scope-10step-2g-$(date +%s)"
  set +e
  output=$(JOB_NAME="$job_name" bash "$FARO_ROOT/scripts/submit_samtok_tb_gppo_predicted_evidence_scope.sh" 2>&1)
  rc=$?
  set -e
  printf '%s\n' "$output" >> "$STATE/submit.log"
  if (( rc == 0 )); then printf '%s\n' "$job_name" > "$marker"; break; fi
  sleep "$INTERVAL"
done
