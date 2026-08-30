#!/usr/bin/env bash
set -euo pipefail

FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
STATE_DIR=${STATE_DIR:-$FARO_ROOT/logs/pv_submit}
INTERVAL=${INTERVAL:-300}
mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/.lock"
flock -n 9 || exit 0

marker="$STATE_DIR/submitted"
while [[ ! -f "$marker" ]]; do
  # A submission may have succeeded from another session while this process
  # was unable to write its marker. Reconcile by prefix before creating a
  # second GPU job; an unavailable API is treated as a transient failure.
  set +e
  existing=$(rjob list --namespace=ailab-dnacoding 2>/dev/null | \
    grep -E 'dna-fepo-paired-view-10step-2g' | head -1)
  existing_status=$?
  set -e
  if (( existing_status == 0 )) && [[ -n "$existing" ]]; then
    printf '%s\n' "$(printf '%s\n' "$existing" | awk '{print $1}')" > "$marker"
    printf '%s existing_job_reconciled\n' "$(date -Is)" >> "$STATE_DIR/submit.log"
    break
  fi
  job_name="dna-fepo-paired-view-10step-2g-$(date +%s)"
  set +e
  output=$(JOB_NAME="$job_name" bash "$FARO_ROOT/scripts/submit_samtok_tb_gppo_paired_view.sh" 2>&1)
  status=$?
  set -e
  printf '%s\n' "$output" >> "$STATE_DIR/submit.log"
  if (( status == 0 )); then
    printf '%s\n' "$job_name" > "$marker"
    break
  fi
  sleep "$INTERVAL"
done
