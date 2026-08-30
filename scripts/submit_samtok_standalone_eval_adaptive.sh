#!/usr/bin/env bash
set -euo pipefail

FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
ADAPTER=${ADAPTER:?ADAPTER is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
JOB_PREFIX=${JOB_PREFIX:-dna-samtok-standalone-eval}
HOLDOUT=${HOLDOUT:-$FARO_ROOT/data/fepo_existence/grefcoco_selective_holdout_256.jsonl}

adapter_real=$(realpath "$ADAPTER")
holdout_real=$(realpath "$HOLDOUT")
output_real=$(realpath -m "$OUTPUT")
ADAPTER="$adapter_real"
HOLDOUT="$holdout_real"
OUTPUT="$output_real"
allowed_root=$(realpath "$FARO_ROOT/outputs/samtok_selective")
case "$adapter_real" in
  "$allowed_root"/*/adapter) ;;
  *) echo "Adapter must come from standalone SAMTok training: $adapter_real" >&2; exit 2 ;;
esac
run_root=$(dirname "$adapter_real")
[[ -f "$run_root/metrics.json" ]] || { echo "Missing training metrics: $run_root/metrics.json" >&2; exit 2; }
[[ -f "$run_root/provenance_manifest.json" ]] || {
  echo "Missing training provenance: $run_root/provenance_manifest.json" >&2; exit 2;
}
python3 - "$run_root/metrics.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "finished":
    raise SystemExit(f"Training run is not finished: {payload.get('status')!r}")
PY
[[ $(wc -l < "$HOLDOUT") -eq 512 ]] || { echo "Holdout must contain exactly 512 rows" >&2; exit 2; }
anchor_adapter=""
eval_adapter="$adapter_real"
if [[ -f "$adapter_real/visual/adapter_config.json" ]]; then
  eval_adapter="$adapter_real/visual"
  anchor_adapter=${SAMTOK_STANDALONE_ANCHOR:-$FARO_ROOT/outputs/samtok_selective/continued_sft_to500/adapter}
  [[ -f "$eval_adapter/adapter_model.safetensors" ]] || {
    echo "Missing visual adapter weights: $eval_adapter" >&2; exit 2;
  }
  [[ -f "$anchor_adapter/adapter_config.json" && -f "$anchor_adapter/adapter_model.safetensors" ]] || {
    echo "Missing frozen SAMTok anchor: $anchor_adapter" >&2; exit 2;
  }
fi
case "$JOB_PREFIX" in dna-*) ;; *) echo "JOB_PREFIX must start with dna-" >&2; exit 2 ;; esac
case "$(realpath -m "$OUTPUT")" in
  "$FARO_ROOT"/evals/*) ;;
  *) echo "OUTPUT must be under $FARO_ROOT/evals" >&2; exit 2 ;;
esac

# Enforce terminal branch decisions at the shared submission boundary. Older
# monitor daemons can survive a code refresh and otherwise submit a closed
# candidate when the control plane returns. These checks are read-only and
# fail closed; a ``closed_training_gate`` PV decision is rejected, PES normal
# remains eligible, while shuffled PES is gated by its completed normal worker
# contract.
CLOSED_EVAL=""
case "$(basename "$OUTPUT")" in
  paired_view_holdout512)
    python3 - "$FARO_ROOT/evals/pv_training_gate.json" <<'PY'
import json
import sys
from pathlib import Path
try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit("paired-view training decision is missing or invalid")
if value.get("decision") != "open":
    raise SystemExit("paired-view training gate is not open")
PY
    ;;
  boundary_bottleneck_paired_view_holdout512)
    CLOSED_EVAL="$FARO_ROOT/evals/boundary_bottleneck_paired_view_vs_matched_sft_bootstrap20k.json"
    ;;
  boundary_stratified_native_rank_local_holdout512)
    CLOSED_EVAL="$FARO_ROOT/evals/boundary_stratified_native_rank_local_vs_matched_sft_bootstrap20k.json"
    ;;
  action_budget_native_rank_local_holdout512)
    CLOSED_EVAL="$FARO_ROOT/evals/action_budget_native_rank_local_vs_matched_sft_bootstrap20k.json"
    ;;
  *)
    CLOSED_EVAL=""
    ;;
esac
if [[ -n "$CLOSED_EVAL" ]]; then
  python3 - "$CLOSED_EVAL" <<'PY'
import json
import sys
from pathlib import Path
try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(0)
if value.get("promotion_gate") is False or value.get("ci_corrected_promotion_gate") is False:
    raise SystemExit("candidate evaluation branch is already closed")
PY
fi
if [[ "$(basename "$OUTPUT")" == predicted_evidence_scope_shuffled_holdout512 ]]; then
  python3 - "$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_10step_2gpu/metrics.json" <<'PY'
import json
import sys
from pathlib import Path
try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit("shuffled PES requires normal PES metrics")
gate = value.get("validity_gate") or {}
ready = (
    value.get("status") == "finished"
    and len(value.get("steps") or []) >= 10
    and gate.get("passed") is True
    and gate.get("effective_support_gate_passed") is True
    and gate.get("tail_risk_gate_passed") is True
    and gate.get("pes_coverage_gate_passed") is True
)
if not ready:
    raise SystemExit("shuffled PES requires a valid normal PES worker")
PY
fi

# A stale monitor version may invoke this entrypoint more frequently than the
# requested five-minute retry window. Serialize by output and reserve one
# 300-second retry slot before contacting rjob, so only one submit attempt can
# package the evaluator even across monitor restarts.
GUARD_ROOT="$FARO_ROOT/logs/eval_submit_guard"
mkdir -p "$GUARD_ROOT"
GUARD_KEY=$(basename "$OUTPUT")
exec 8>"$GUARD_ROOT/${GUARD_KEY}.lock"
flock -n 8 || exit 75
GUARD_RETRY="$GUARD_ROOT/${GUARD_KEY}.retry_after"
now=$(date +%s)
if [[ -s "$GUARD_RETRY" ]]; then
  deadline=$(sed -n '1p' "$GUARD_RETRY" 2>/dev/null || true)
  if [[ "$deadline" =~ ^[0-9]+$ ]] && (( now < deadline )); then
    exit 75
  fi
fi
printf '%s\n' "$((now + 300))" > "$GUARD_RETRY"

set +e
ADAPTER="$eval_adapter" \
ANCHOR_ADAPTER="$anchor_adapter" \
OUTPUT="$OUTPUT" \
JOB_PREFIX="$JOB_PREFIX" \
POLL_SECONDS="${POLL_SECONDS:-300}" \
REFSEG_SCHEMA="$HOLDOUT" \
GEOMETRY_SCHEMA=none \
MASKCAP_SCHEMA=none \
EXISTENCE_SCHEMA=none \
bash "$FARO_ROOT/scripts/submit_fepo_eval_adaptive.sh"
status=$?
set -e
exit "$status"
