#!/usr/bin/env bash
set -euo pipefail

FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=${OUT:-$FARO_ROOT/evals}
STATE=${STATE:-$FARO_ROOT/logs/r35_submit}
INTERVAL=${INTERVAL:-300}
mkdir -p "$STATE"
exec 9>"$STATE/.lock"
flock -n 9 || exit 0

decision_ready() {
  python3 - \
    "$OUT/paired_view_vs_r18_bootstrap20k.json" \
    "$OUT/paired_view_vs_matched_sft_bootstrap20k.json" \
    "$OUT/paired_view_vs_r18_slices20k.json" \
    "$OUT/pv_training_gate.json" \
    "$OUT/r18_matched_sft_vs_r18_bootstrap20k.json" <<'PY'
import json
import sys
from pathlib import Path

pv_r18, pv_sft, pv_slices, pv_decision, sft_r18 = [Path(x) for x in sys.argv[1:]]
if not sft_r18.is_file() or sft_r18.stat().st_size == 0:
    raise SystemExit(1)
if pv_decision.is_file() and pv_decision.stat().st_size > 0:
    decision = json.loads(pv_decision.read_text(encoding="utf-8"))
    if decision.get("decision") != "closed_training_gate":
        raise SystemExit(1)
    if float(decision.get("joint_positive_fraction_mean", 1.0)) >= 0.20:
        raise SystemExit(1)
    value = json.loads(sft_r18.read_text(encoding="utf-8"))
    if int(value.get("num_paired", 0)) != 512:
        raise SystemExit(1)
    if int((value.get("selective_utility_delta") or {}).get("bootstrap_repeats", 0)) != 20000:
        raise SystemExit(1)
    raise SystemExit(0)
paths = [pv_r18, pv_sft, pv_slices]
if not all(p.is_file() and p.stat().st_size > 0 for p in paths):
    raise SystemExit(1)
try:
    values = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
except (OSError, ValueError):
    raise SystemExit(1)
for value in values[:2]:
    if int(value.get("num_paired", 0)) != 512:
        raise SystemExit(1)
    for key in ("positive_ciou_delta", "selective_utility_delta", "no_target_explicit_recall_delta"):
        metric = value.get(key) or {}
        if int(metric.get("bootstrap_repeats", 0)) != 20000:
            raise SystemExit(1)
    if bool(value.get("ci_corrected_promotion_gate", value.get("promotion_gate", True))):
        raise SystemExit(1)
slice_report = values[2]
if int(slice_report.get("num_paired", 0)) != 512:
    raise SystemExit(1)
if not bool(slice_report.get("slice_gate", False)):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

marker="$STATE/submitted"
while [[ ! -f "$marker" ]]; do
  if ! decision_ready; then
    sleep "$INTERVAL"
    continue
  fi
  # Probe the namespace before invoking the heavyweight submit wrapper.  A
  # DNS/API outage must not repeatedly package the worker image and hold the
  # submission lock while no rjob can possibly be created.
  set +e
  rjob list --namespace=ailab-dnacoding >/dev/null 2>&1
  control_status=$?
  set -e
  if (( control_status != 0 )); then
    printf '%s control_plane_unavailable status=%s\n' "$(date -Is)" "$control_status" >> "$STATE/submit.log"
    sleep "$INTERVAL"
    continue
  fi
  existing=$(rjob list --namespace=ailab-dnacoding 2>/dev/null | grep -E 'dna-fepo-safe-visual-interface-10step-2g' | head -1 || true)
  if [[ -n "$existing" ]]; then
    printf '%s\n' "$(printf '%s\n' "$existing" | awk '{print $1}')" > "$marker"
    break
  fi
  job_name="dna-fepo-safe-visual-interface-10step-2g-$(date +%s)"
  set +e
  output=$(JOB_NAME="$job_name" bash "$FARO_ROOT/scripts/submit_samtok_tb_gppo_safe_visual_interface.sh" 2>&1)
  status=$?
  set -e
  printf '%s\n' "$output" >> "$STATE/submit.log"
  if (( status == 0 )); then
    printf '%s\n' "$job_name" > "$marker"
    break
  fi
  sleep "$INTERVAL"
done
