#!/usr/bin/env bash
set -euo pipefail

# Idempotent transition into the isolated AB-FEPO screen. The runner waits for
# all preceding visual/boundary branches to close and retries only the control
# plane query; it never bypasses the marker or creates a duplicate job.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
OUT=${OUT:-$FARO_ROOT/outputs/samtok_selective}
STATE=${STATE:-$FARO_ROOT/logs/ab_submit}
INTERVAL=${INTERVAL:-300}
mkdir -p "$STATE"
exec 9>"$STATE/.lock"
flock -n 9 || exit 0

closed_metrics() {
  python3 - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checks = {
    "fepo_tb_gppo_plain_rank_unified_safe_visual_interface_10step_2gpu":
        ("failed_validity_gate", False),
    "fepo_tb_gppo_plain_rank_unified_boundary_bottleneck_paired_view_10step_2gpu":
        ("finished", True),
    "fepo_tb_gppo_plain_rank_unified_boundary_stratified_native_rank_local_10step_2gpu":
        ("finished", True),
}
for name, (status, gate) in checks.items():
    path = root / name / "metrics.json"
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(1)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SystemExit(1)
    validity = payload.get("validity_gate") or {}
    if payload.get("status") != status or validity.get("passed") is not gate:
        raise SystemExit(1)
    if len(payload.get("steps") or []) < 10:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

marker="$STATE/submitted"
while [[ ! -s "$marker" ]]; do
  if ! closed_metrics; then
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
  existing=$(printf '%s\n' "$list_output" | grep -E 'dna-fepo-action-budget-native-rank-local-10step-2g' | head -1 || true)
  if [[ -n "$existing" ]]; then
    printf '%s\n' "$(printf '%s\n' "$existing" | awk '{print $1}')" > "$marker"
    break
  fi
  job_name="dna-fepo-action-budget-native-rank-local-10step-2g-$(date +%s)"
  set +e
  output=$(JOB_NAME="$job_name" bash "$FARO_ROOT/scripts/submit_samtok_tb_gppo_action_budget_native_rank_local.sh" 2>&1)
  status=$?
  set -e
  printf '%s\n' "$output" >> "$STATE/submit.log"
  if (( status == 0 )); then
    printf '%s\n' "$job_name" > "$marker"
    break
  fi
  sleep "$INTERVAL"
done
