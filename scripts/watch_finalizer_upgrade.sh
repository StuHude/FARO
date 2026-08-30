#!/usr/bin/env bash
set -euo pipefail

# Wait for an older finalizer instance to release its lock, then hand control
# to the current state machine. This never removes or bypasses a lock.
FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
STATE=${STATE:-$FARO_ROOT/logs/screen_monitor/finalize}
# Older deployments used versioned state directories. Wait for every known
# finalizer lock so a patched state machine cannot run beside stale logic.
LOCKS=(
  "$FARO_ROOT/logs/screen_monitor/finalize/.lock"
  "$FARO_ROOT/logs/screen_monitor/finalize_v8/.lock"
)
mkdir -p "$STATE"
while :; do
  all_free=1
  for lock in "${LOCKS[@]}"; do
    [[ -e "$lock" ]] || continue
    if ! flock -n "$lock" -c true 2>/dev/null; then
      all_free=0
      break
    fi
  done
  (( all_free == 1 )) && break
  sleep 60
done
exec bash "$FARO_ROOT/scripts/monitor_finalize_matched_sft_pv.sh"
