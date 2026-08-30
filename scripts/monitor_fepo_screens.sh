#!/usr/bin/env bash
set -euo pipefail

# Long-lived screen monitor. Training jobs remain queued independently; once a
# candidate has a finished metrics contract, run its full adaptive evaluation.
FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT_ROOT=${OUT_ROOT:-$FARO_ROOT/evals}
LOG_ROOT=${LOG_ROOT:-$FARO_ROOT/logs/screen_monitor}
INTERVAL=${INTERVAL:-60}
RETRY_SECONDS=${RETRY_SECONDS:-300}
mkdir -p "$LOG_ROOT" "$OUT_ROOT"

# Multiple shells may restart the monitor after a session reconnect. Keep one
# scheduler loop so a finished candidate cannot trigger duplicate eval jobs.
exec 9>"${MONITOR_LOCK_FILE:-$LOG_ROOT/.monitor.lock.v3}"
flock -n 9 || exit 0

declare -a CANDIDATES=(
  "native_rank_local|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_native_rank_local_10step_2gpu"
  "scale_stratified_native_rank_local|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_scale_stratified_native_rank_local_10step_2gpu"
  "bidirectional_coarse_fine|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_bidirectional_coarse_fine_10step_2gpu"
  "anchor_kl|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_anchor_kl_10step_2gpu"
  "uncertainty_native_rank_local|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_uncertainty_native_rank_local_10step_2gpu"
  "conservative_null_tail|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_conservative_null_tail_10step_2gpu"
  "confidence_gated_native_rank_local|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_confidence_gated_native_rank_local_10step_2gpu"
  "margin_calibrated_native_rank_local|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_margin_calibrated_native_rank_local_10step_2gpu"
  "primal_dual_null_risk|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_primal_dual_null_risk_10step_2gpu"
  "grounded_interface|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_grounded_interface_10step_2gpu"
)

status_finished() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "finished" else 1)
PY
}

contract_finished() {
  python3 - "$1" "$(dirname "$1")/provenance_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    provenance = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
method = payload.get("method") or {}
initialization = payload.get("initialization") or {}
validity = payload.get("validity_gate") or {}
data = provenance.get("data") or {}
checks = (
    payload.get("status") == "finished",
    int(payload.get("steps_completed", 0)) >= 10,
    int(payload.get("optimizer_updates_completed", 0)) > 0,
    int(data.get("row_count", 0)) >= 5120,
    int(method.get("rollouts_per_prompt", 0)) >= 4,
    "samtok" in str(initialization.get("path", "")).lower(),
    validity.get("passed") is True,
    validity.get("effective_support_gate_passed") is True,
    validity.get("tail_risk_gate_passed") is True,
)
raise SystemExit(0 if all(checks) else 1)
PY
}

while :; do
  heartbeat=""
  for entry in "${CANDIDATES[@]}"; do
    name=${entry%%|*}
    run_root=${entry#*|}
    metrics="$run_root/metrics.json"
    marker="$LOG_ROOT/${name}.eval_submitted"
    retry_after="$LOG_ROOT/${name}.eval_retry_after"
    output="$OUT_ROOT/${name}_holdout512"
    if [[ -f "$marker" ]]; then
      marker_state=$(sed -n '1p' "$marker" 2>/dev/null || true)
      if [[ "$marker_state" != "PENDING_CONTROL_PLANE" ]]; then
        heartbeat+=" ${name}=eval_submitted"
        continue
      fi
    fi
    if [[ -f "$output" ]]; then
      heartbeat+=" ${name}=eval_finished"
      continue
    fi
    if [[ -s "$retry_after" ]]; then
      retry_deadline=$(sed -n '1p' "$retry_after" 2>/dev/null || true)
      now=$(date +%s)
      if [[ "$retry_deadline" =~ ^[0-9]+$ ]] && (( now < retry_deadline )); then
        heartbeat+=" ${name}=retry_backoff"
        continue
      fi
      rm -f "$retry_after"
    fi
    if ! status_finished "$metrics" 2>/dev/null; then
      heartbeat+=" ${name}=waiting"
      continue
    fi
    if ! contract_finished "$metrics" 2>/dev/null; then
      heartbeat+=" ${name}=finished_contract_failed"
      continue
    fi
    adapter="$run_root/adapter"
    # Representation runs save the trainable adapter under ``visual/`` while
    # language-only runs save it at the adapter root.
    if [[ -f "$adapter/visual/adapter_config.json" && -f "$adapter/visual/adapter_model.safetensors" ]]; then
      adapter="$adapter"
    elif [[ ! -f "$adapter/adapter_config.json" || ! -f "$adapter/adapter_model.safetensors" ]]; then
      heartbeat+=" ${name}=finished_missing_adapter"
      continue
    fi
    heartbeat+=" ${name}=eval_submit"
    printf '%s candidate=%s status=finished\n' "$(date -Is)" "$name" >> "$LOG_ROOT/monitor.log"
    printf '%s\n' PENDING_CONTROL_PLANE > "$marker"
    # Reserve the retry window before touching the control plane.  This keeps
    # concurrent/restarted monitors from issuing one submit attempt per loop
    # when the API is unavailable before an rjob is created.
    printf '%s\n' "$(( $(date +%s) + RETRY_SECONDS ))" > "$retry_after"
    (
      if ADAPTER="$adapter" \
        OUTPUT="$OUT_ROOT/${name}_holdout512" \
        JOB_PREFIX="dna-fepo-${name}-eval" \
        POLL_SECONDS=300 \
        bash "$FARO_ROOT/scripts/submit_samtok_standalone_eval_adaptive.sh" \
        >> "$LOG_ROOT/${name}.eval.log" 2>&1; then
          printf '%s\n' SUBMITTED > "$marker"
          rm -f "$retry_after"
        else
          printf '%s\n' PENDING_CONTROL_PLANE > "$marker"
          printf '%s\n' "$(( $(date +%s) + RETRY_SECONDS ))" > "$retry_after"
          printf '%s candidate=%s eval_submit_failed\n' "$(date -Is)" "$name" >> "$LOG_ROOT/monitor.log"
        fi
    ) &
  done
  printf '%s heartbeat%s\n' "$(date -Is)" "$heartbeat" >> "$LOG_ROOT/monitor.log"
  sleep "$INTERVAL"
done
