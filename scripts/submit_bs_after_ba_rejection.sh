#!/usr/bin/env bash
set -euo pipefail

# Idempotent transition: BA must have a complete rejected holdout before BS
# can consume any GPU. The submitter itself is lock-protected and never tunes
# the boundary mixture after holdout access.
FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BA_EVAL=${BA_EVAL:-$FARO_ROOT/evals/boundary_bottleneck_paired_view_vs_matched_sft_bootstrap20k.json}
STATE=${STATE:-$FARO_ROOT/logs/bs_submit}
INTERVAL=${INTERVAL:-300}
mkdir -p "$STATE"
exec 9>"$STATE/.lock"
flock -n 9 || exit 0

ba_rejected() {
  python3 - "$BA_EVAL" <<'PY'
import json, sys
from pathlib import Path
try:
    d=json.loads(Path(sys.argv[1]).read_text())
except (OSError, ValueError):
    raise SystemExit(1)
g=d.get("promotion_gates") or d.get("gates")
if isinstance(g, dict):
    failed = any(v is False for v in g.values())
else:
    # The canonical paired bootstrap writer stores the aggregate decision as
    # ``promotion_gate`` (and the corrected gate as a sibling boolean).
    failed = d.get("promotion_gate") is False or d.get("ci_corrected_promotion_gate") is False
if not ("promotion_gate" in d or "ci_corrected_promotion_gate" in d or g):
    raise SystemExit(1)
print(json.dumps({"complete": True, "rejected": failed}, sort_keys=True))
raise SystemExit(0 if failed else 1)
PY
}

marker="$STATE/submitted"
while [[ ! -s "$marker" ]]; do
  ba_rejected || { sleep "$INTERVAL"; continue; }
  set +e
  listing=$(rjob list --namespace=ailab-dnacoding 2>&1)
  rc=$?
  set -e
  if (( rc != 0 )); then
    printf '%s control_plane_unavailable status=%s\n' "$(date -Is)" "$rc" >> "$STATE/submit.log"
    sleep "$INTERVAL"; continue
  fi
  existing=$(printf '%s\n' "$listing" | grep -E 'dna-fepo-boundary-stratified-native-rank-local-10step-2g' | head -1 || true)
  if [[ -n "$existing" ]]; then
    printf '%s\n' "$(printf '%s\n' "$existing" | awk '{print $1}')" > "$marker"; break
  fi
  job_name="dna-fepo-boundary-stratified-native-rank-local-10step-2g-$(date +%s)"
  set +e
  output=$(JOB_NAME="$job_name" bash "$FARO_ROOT/scripts/submit_samtok_tb_gppo_boundary_stratified_native_rank_local.sh" 2>&1)
  rc=$?
  set -e
  printf '%s\n' "$output" >> "$STATE/submit.log"
  if (( rc == 0 )); then printf '%s\n' "$job_name" > "$marker"; break; fi
  sleep "$INTERVAL"
done
