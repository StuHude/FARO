#!/usr/bin/env bash
set -euo pipefail

# Watches candidates registered after the long-lived R21-R28 monitor.  A
# separate lock keeps this helper independent while it owns only late-screen
# markers (R29/R30/R35) and never touches training jobs.
FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT_ROOT=${OUT_ROOT:-$FARO_ROOT/evals}
LOG_ROOT=${LOG_ROOT:-$FARO_ROOT/logs/screen_monitor}
INTERVAL=${INTERVAL:-60}
RETRY_SECONDS=${RETRY_SECONDS:-300}
mkdir -p "$LOG_ROOT" "$OUT_ROOT"
exec 9>"${MONITOR_LOCK_FILE:-$LOG_ROOT/.late_monitor.lock.v3}"
flock -n 9 || exit 0

declare -a CANDIDATES=(
  "grounded_interface|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_grounded_interface_10step_2gpu"
  "safe_visual_interface|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_safe_visual_interface_10step_2gpu"
  "paired_view|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_paired_view_10step_2gpu"
  # BA is a post-PV candidate.  This entry only enables automatic evaluation
  # after a separately gated training submission; it never submits training.
  "boundary_bottleneck_paired_view|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_boundary_bottleneck_paired_view_10step_2gpu"
  "boundary_stratified_native_rank_local|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_boundary_stratified_native_rank_local_10step_2gpu"
  "action_budget_native_rank_local|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_action_budget_native_rank_local_10step_2gpu"
  "predicted_evidence_scope|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_10step_2gpu"
  "predicted_evidence_scope_shuffled|$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_shuffled_10step_2gpu"
  # Matched-budget continued-SFT control for the selected R18 policy.  It is
  # evaluated with the same 512-row schema and adaptive GPU ladder; no RL
  # claim is made until this control is paired against R18.
  "r18_matched_sft|$FARO_ROOT/outputs/samtok_selective/continued_sft_r18_matched_200"
)

status_finished() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path
try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
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
    metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    provenance = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
method = metrics.get("method") or {}
init = metrics.get("initialization") or {}
validity = metrics.get("validity_gate") or {}
data = provenance.get("data") or {}
checks = (
    metrics.get("status") == "finished",
    int(metrics.get("steps_completed", 0)) >= 10,
    int(metrics.get("optimizer_updates_completed", 0)) > 0,
    int(data.get("row_count", 0)) >= 5120,
    int(method.get("rollouts_per_prompt", 0)) >= 4,
    "samtok" in str(init.get("path", "")).lower(),
    validity.get("passed") is True,
    validity.get("effective_support_gate_passed") is True,
    validity.get("tail_risk_gate_passed") is True,
)
if "predicted_evidence_scope" in str(provenance.get("stage", "")):
    checks = checks + (validity.get("pes_coverage_gate_passed") is True,)
raise SystemExit(0 if all(checks) else 1)
PY
}

# A date-only runner_started marker can outlive a killed shell.  Require a
# live process whose command line names the transition script before treating
# the marker as active; the transition's flock remains the final idempotency
# guard if two monitors race during recovery.
runner_active() {
  local state=$1 script=$2 pid
  [[ -s "$state/pid" ]] || return 1
  pid=$(sed -n '1p' "$state/pid" 2>/dev/null || true)
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  ps -p "$pid" -o args= 2>/dev/null | grep -Fq "$script"
}

sft_contract_finished() {
  python3 - "$1" "$(dirname "$1")/provenance_manifest.json" <<'PY'
import json
import sys
from pathlib import Path
try:
    metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    provenance = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
data = provenance.get("data") or {}
checks = (
    metrics.get("status") == "finished",
    int(metrics.get("steps_completed", 0)) >= 10,
    len(metrics.get("steps") or []) >= 10,
    int(data.get("row_count", 0)) >= 5120,
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
    if [[ "$name" == "paired_view" && -s "$OUT_ROOT/pv_training_gate.json" ]] && \
      grep -q '"decision": "closed_training_gate"' "$OUT_ROOT/pv_training_gate.json"; then
      heartbeat+=" ${name}=closed_training_gate"
      continue
    fi
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
    if [[ "$name" == "r18_matched_sft" ]]; then
      finished_check=sft_contract_finished
    else
      finished_check=contract_finished
    fi
    if ! "$finished_check" "$metrics" 2>/dev/null; then
      heartbeat+=" ${name}=finished_contract_failed"
      continue
    fi
    adapter="$run_root/adapter"
    # R30's representation adapter is nested under ``visual/``; pass the
    # parent directory to the evaluator so it can compose the frozen anchor.
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
  # R35's worker validity failure is the only condition that unlocks BA-FEPO.
  # The submitter owns its lock and 300-second control-plane retry loop.
  BA_STATE="$FARO_ROOT/logs/ba_submit"
  if [[ ! -f "$BA_STATE/runner_started" ]] &&
     python3 - "$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_safe_visual_interface_10step_2gpu/metrics.json" <<'PY'
import json
import sys
from pathlib import Path
try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
gate = value.get("validity_gate") or {}
raise SystemExit(0 if value.get("status") == "failed_validity_gate" and gate.get("active_set_risk_gate_passed") is False and gate.get("tail_risk_gate_passed") is True else 1)
PY
  then
    mkdir -p "$BA_STATE"
    printf '%s\n' "$(date -Is)" > "$BA_STATE/runner_started"
    (STATE="$BA_STATE" bash "$FARO_ROOT/scripts/submit_ba_after_r35_failure.sh" >> "$BA_STATE/runner.log" 2>&1) &
    heartbeat+=" ba_submit=started"
  fi
  # BA's complete holdout rejection is the only condition that unlocks the
  # isolated boundary-stratified sampling arm. The transition owns its lock
  # and performs its own authenticated queue/idempotency checks.
  BS_STATE="$FARO_ROOT/logs/bs_submit"
  if [[ ! -f "$BS_STATE/runner_started" ]] &&
     [[ -s "$FARO_ROOT/evals/boundary_bottleneck_paired_view_vs_matched_sft_bootstrap20k.json" ]] &&
     python3 - "$FARO_ROOT/evals/boundary_bottleneck_paired_view_vs_matched_sft_bootstrap20k.json" <<'PY'
import json, sys
from pathlib import Path
try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
rejected = value.get("promotion_gate") is False or value.get("ci_corrected_promotion_gate") is False
raise SystemExit(0 if rejected else 1)
PY
  then
    mkdir -p "$BS_STATE"
    printf '%s\n' "$(date -Is)" > "$BS_STATE/runner_started"
    (STATE="$BS_STATE" bash "$FARO_ROOT/scripts/submit_bs_after_ba_rejection.sh" >> "$BS_STATE/runner.log" 2>&1) &
    heartbeat+=" bs_submit=started"
  fi
  # AB-FEPO is unlocked only by a complete, rejected BS holdout. Its
  # transition owns a separate lock and never overlaps another training arm.
  AB_STATE="$FARO_ROOT/logs/ab_submit"
  if [[ ! -f "$AB_STATE/runner_started" ]] &&
     [[ -s "$FARO_ROOT/evals/boundary_stratified_native_rank_local_vs_matched_sft_bootstrap20k.json" ]] &&
     python3 - "$FARO_ROOT/evals/boundary_stratified_native_rank_local_vs_matched_sft_bootstrap20k.json" <<'PY'
import json, sys
from pathlib import Path
try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
rejected = value.get("promotion_gate") is False or value.get("ci_corrected_promotion_gate") is False
raise SystemExit(0 if rejected else 1)
PY
  then
    mkdir -p "$AB_STATE"
    printf '%s\n' "$(date -Is)" > "$AB_STATE/runner_started"
    (STATE="$AB_STATE" bash "$FARO_ROOT/scripts/submit_ab_after_screen_closure.sh" >> "$AB_STATE/runner.log" 2>&1) &
    heartbeat+=" ab_submit=started"
  fi
  # PES-FEPO is unlocked only by the complete, rejected AB holdout.
  PES_STATE="$FARO_ROOT/logs/pes_submit"
  if [[ ! -s "$PES_STATE/submitted" ]] &&
     { [[ ! -f "$PES_STATE/runner_started" ]] ||
       ! runner_active "$PES_STATE" "submit_pes_after_ab_rejection.sh"; } &&
     [[ -s "$FARO_ROOT/evals/action_budget_native_rank_local_vs_matched_sft_bootstrap20k.json" ]] &&
     python3 - "$FARO_ROOT/evals/action_budget_native_rank_local_vs_matched_sft_bootstrap20k.json" <<'PY'
import json, sys
from pathlib import Path
try:
    d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if d.get("promotion_gate") is False or d.get("ci_corrected_promotion_gate") is False else 1)
PY
  then
    mkdir -p "$PES_STATE"
    printf '%s\n' "$(date -Is)" > "$PES_STATE/runner_started"
    (STATE="$PES_STATE" bash "$FARO_ROOT/scripts/submit_pes_after_ab_rejection.sh" >> "$PES_STATE/runner.log" 2>&1) &
    heartbeat+=" pes_submit=started"
  fi
  PES_SHUFFLED_STATE="$FARO_ROOT/logs/pes_shuffled_submit"
  if { [[ ! -f "$PES_SHUFFLED_STATE/runner_started" ]] ||
       ! runner_active "$PES_SHUFFLED_STATE" "submit_pes_shuffled_after_pes_completion.sh"; } &&
     [[ -s "$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_10step_2gpu/metrics.json" ]] &&
     python3 - "$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_10step_2gpu/metrics.json" <<'PY'
import json, sys
from pathlib import Path
try:
    d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
gate = d.get("validity_gate") or {}
ready = (
    d.get("status") == "finished"
    and len(d.get("steps") or []) >= 10
    and gate.get("passed") is True
    and gate.get("effective_support_gate_passed") is True
    and gate.get("tail_risk_gate_passed") is True
    and gate.get("pes_coverage_gate_passed") is True
)
raise SystemExit(0 if ready else 1)
PY
  then
    mkdir -p "$PES_SHUFFLED_STATE"
    printf '%s\n' "$(date -Is)" > "$PES_SHUFFLED_STATE/runner_started"
    (STATE="$PES_SHUFFLED_STATE" bash "$FARO_ROOT/scripts/submit_pes_shuffled_after_pes_completion.sh" >> "$PES_SHUFFLED_STATE/runner.log" 2>&1) &
    heartbeat+=" pes_shuffled_submit=started"
  fi
  # Once both PES evaluators finish, finalize all paired 512-row/20k reports
  # under a separate lock. A failed local analysis is retried on the next
  # heartbeat; no training or evaluator job is submitted by this transition.
  PES_FINALIZE_STATE="$FARO_ROOT/logs/pes_finalize"
  if [[ ! -s "$PES_FINALIZE_STATE/decision.json" ]] &&
     [[ -s "$OUT_ROOT/predicted_evidence_scope_holdout512" ]] &&
     [[ -s "$OUT_ROOT/predicted_evidence_scope_shuffled_holdout512" ]] &&
     [[ ! -f "$PES_FINALIZE_STATE/running" ]]; then
    mkdir -p "$PES_FINALIZE_STATE"
    printf '%s\n' "$(date -Is)" > "$PES_FINALIZE_STATE/running"
    heartbeat+=" pes_finalize=started"
    (
      if bash "$FARO_ROOT/scripts/finalize_pes_eval.sh" >> "$PES_FINALIZE_STATE/finalize.log" 2>&1; then
        rm -f "$PES_FINALIZE_STATE/running"
      else
        rm -f "$PES_FINALIZE_STATE/running"
        printf '%s pes_finalize_failed\n' "$(date -Is)" >> "$LOG_ROOT/monitor.log"
      fi
    ) &
  fi
  printf '%s late-heartbeat%s\n' "$(date -Is)" "$heartbeat" >> "$LOG_ROOT/monitor.log"
  sleep "$INTERVAL"
done
