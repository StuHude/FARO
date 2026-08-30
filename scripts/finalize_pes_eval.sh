#!/usr/bin/env bash
set -euo pipefail

FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EVAL_ROOT=${EVAL_ROOT:-$FARO_ROOT/evals}
LOG_ROOT=${LOG_ROOT:-$FARO_ROOT/logs/pes_finalize}
BASE=${BASE:-$EVAL_ROOT/r18_100_confirmation_holdout512}
SFT=${SFT:-$EVAL_ROOT/r18_matched_sft_holdout512}
NORMAL=${NORMAL:-$EVAL_ROOT/predicted_evidence_scope_holdout512}
SHUFFLED=${SHUFFLED:-$EVAL_ROOT/predicted_evidence_scope_shuffled_holdout512}
REPEATS=${REPEATS:-20000}
mkdir -p "$EVAL_ROOT" "$LOG_ROOT"
exec 9>"$LOG_ROOT/.lock"
flock -n 9 || exit 0

for path in "$BASE" "$SFT" "$NORMAL" "$SHUFFLED"; do
  [[ -s "$path" ]] || { printf 'waiting for %s\n' "$path" >&2; exit 3; }
done

PYTHONPATH="$FARO_ROOT" python3 - "$BASE" "$SFT" "$NORMAL" "$SHUFFLED" <<'PY'
import json
import sys
from pathlib import Path

paths = [Path(value) for value in sys.argv[1:]]
bundles = []
for path in paths:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["refseg_overall"]["records"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"invalid PES evaluation {path}: {exc}")
    if len(rows) != 512:
        raise SystemExit(f"{path}: expected 512 rows, got {len(rows)}")
    ids = [str(row.get("id")) for row in rows]
    if len(set(ids)) != 512:
        raise SystemExit(f"{path}: IDs are not unique")
    positives = sum(bool(row.get("truth_exists")) for row in rows)
    if positives != 256:
        raise SystemExit(f"{path}: expected 256 positive rows, got {positives}")
    for row in rows:
        if "boundary_iou" not in row:
            raise SystemExit(f"{path}: missing boundary_iou for {row.get('id')}")
        if not (bool(row.get("valid_mask_tokens")) or bool(row.get("explicit_null"))):
            raise SystemExit(f"{path}: invalid output for {row.get('id')}")
    bundles.append({str(row["id"]): bool(row["truth_exists"]) for row in rows})
if any(bundle != bundles[0] for bundle in bundles[1:]):
    raise SystemExit("PES evaluations do not share identical IDs/truth labels")
PY

run_pair() {
  local name=$1 left=$2 right=$3 seed=$4
  local output="$EVAL_ROOT/${name}_bootstrap20k.json"
  if [[ ! -s "$output" ]]; then
    PYTHONPATH="$FARO_ROOT" python3 "$FARO_ROOT/tools/analyze_selective_eval.py" \
      "$left" "$right" --section refseg_overall --repeats "$REPEATS" --seed "$seed" \
      --noninferiority-margin 0.01 --min-utility-ci-lower 0.0 \
      --require-utility-ci-positive --min-positive-ciou-ci-lower -0.01 \
      --min-negative-ci-lower -0.01 --output "$output"
  fi
  local slices="$EVAL_ROOT/${name}_slices20k.json"
  if [[ ! -s "$slices" ]]; then
    PYTHONPATH="$FARO_ROOT" python3 "$FARO_ROOT/tools/analyze_selective_slices.py" \
      "$left" "$right" --section refseg_overall --repeats "$REPEATS" \
      --seed "$seed" --max-drop 0.01 --min-noninferior-slices 2 --output "$slices"
  fi
}

run_pair pes_vs_r18 "$BASE" "$NORMAL" 42
run_pair pes_vs_matched_sft "$SFT" "$NORMAL" 43
run_pair shuffled_pes_vs_r18 "$BASE" "$SHUFFLED" 44
run_pair shuffled_pes_vs_matched_sft "$SFT" "$SHUFFLED" 45
run_pair pes_vs_shuffled "$SHUFFLED" "$NORMAL" 46

PYTHONPATH="$FARO_ROOT" python3 - "$LOG_ROOT/decision.json.tmp" "$EVAL_ROOT" "$REPEATS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out, root, repeats = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
names = (
    "pes_vs_r18", "pes_vs_matched_sft", "shuffled_pes_vs_r18",
    "shuffled_pes_vs_matched_sft", "pes_vs_shuffled",
)
reports = {}
for name in names:
    path = root / f"{name}_bootstrap20k.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("num_paired") != 512:
        raise SystemExit(f"{path}: paired count is not 512")
    if payload.get("selective_utility_delta", {}).get("bootstrap_repeats") != repeats:
        raise SystemExit(f"{path}: bootstrap repeat count mismatch")
    reports[name] = {
        # Keep the historical gate for audit, but never use it for a new
        # promotion decision: its legacy branch does not require a positive
        # utility CI lower bound.
        "legacy_promotion_gate": bool(payload.get("promotion_gate")),
        "ci_corrected_promotion_gate": bool(payload.get("ci_corrected_promotion_gate")),
        "utility_delta": payload.get("selective_utility_delta", {}).get("mean"),
        "positive_ciou_delta": payload.get("positive_ciou_delta", {}).get("mean"),
        "report": str(path),
        "slices": str(root / f"{name}_slices20k.json"),
    }
decision = {
    "status": "finished",
    "method": "PES-FEPO predicted-evidence scope",
    "bootstrap_repeats": repeats,
    "holdout_rows": 512,
    "positive_rows": 256,
    "no_target_rows": 256,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "comparisons": reports,
    # Canonical promotion is CI-corrected for both references.  The explicit
    # legacy field above remains available but cannot promote a candidate.
    "promotion_gate": reports["pes_vs_r18"]["ci_corrected_promotion_gate"] and reports["pes_vs_matched_sft"]["ci_corrected_promotion_gate"],
    "ci_corrected_promotion_gate": reports["pes_vs_r18"]["ci_corrected_promotion_gate"] and reports["pes_vs_matched_sft"]["ci_corrected_promotion_gate"],
}
out.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
mv "$LOG_ROOT/decision.json.tmp" "$LOG_ROOT/decision.json"
printf 'PES finalization complete: %s\n' "$LOG_ROOT/decision.json"
