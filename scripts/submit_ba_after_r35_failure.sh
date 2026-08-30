#!/usr/bin/env bash
set -euo pipefail

# BA-FEPO is eligible only after R35 fails its worker validity gate.  Keep this
# transition idempotent and lock-protected so transient API failures cannot
# create duplicate training jobs.
FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=${OUT:-$FARO_ROOT/outputs/samtok_selective}
R35_METRICS=${R35_METRICS:-$OUT/fepo_tb_gppo_plain_rank_unified_safe_visual_interface_10step_2gpu/metrics.json}
STATE=${STATE:-$FARO_ROOT/logs/ba_submit}
INTERVAL=${INTERVAL:-300}
mkdir -p "$STATE"
exec 9>"$STATE/.lock"
flock -n 9 || exit 0

decision_ready() {
  python3 - "$R35_METRICS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)
try:
    value = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
gate = value.get("validity_gate") or {}
if value.get("status") != "failed_validity_gate":
    raise SystemExit(1)
if gate.get("passed") is not False or gate.get("active_set_risk_gate_passed") is not False:
    raise SystemExit(1)
if len(value.get("steps") or []) < 10:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

marker="$STATE/submitted"
while [[ ! -s "$marker" ]]; do
  if ! decision_ready; then
    sleep "$INTERVAL"
    continue
  fi
  set +e
  list_output=$(rjob list --namespace=ailab-dnacoding 2>&1)
  control_status=$?
  set -e
  if (( control_status != 0 )); then
    printf '%s control_plane_unavailable status=%s\n' "$(date -Is)" "$control_status" >> "$STATE/submit.log"
    sleep "$INTERVAL"
    continue
  fi
  existing=$(printf '%s\n' "$list_output" | grep -E 'dna-fepo-boundary-bottleneck-paired-view-10step-2g' | head -1 || true)
  if [[ -n "$existing" ]]; then
    printf '%s\n' "$(printf '%s\n' "$existing" | awk '{print $1}')" > "$marker"
    break
  fi
  job_name="dna-fepo-boundary-bottleneck-paired-view-10step-2g-$(date +%s)"
  set +e
  output=$(JOB_NAME="$job_name" bash "$FARO_ROOT/scripts/submit_samtok_tb_gppo_boundary_bottleneck_paired_view.sh" 2>&1)
  status=$?
  set -e
  printf '%s\n' "$output" >> "$STATE/submit.log"
  if (( status == 0 )); then
    printf '%s\n' "$job_name" > "$marker"
    break
  fi
  sleep "$INTERVAL"
done
