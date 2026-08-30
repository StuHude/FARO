#!/usr/bin/env bash
set -euo pipefail

# One-shot postprocessor for the historical R16 protocol-alignment eval.
# It waits for the evaluator's complete JSON artifact, then runs the same
# 20,000-paired bootstrap and fixed slice gate used by candidate screens.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
EVAL=${EVAL:-}
BASE=${BASE:-$FARO_ROOT/evals/r18_depth_local_rarity_free_seed17_holdout512_diagnostic}
OUT=${OUT:-$FARO_ROOT/evals/r16_depth_local_rarity_free_holdout512_enhanced_analysis.json}
SLICES=${SLICES:-$FARO_ROOT/evals/r16_depth_local_rarity_free_holdout512_enhanced_slices.json}
LOG=${LOG:-$FARO_ROOT/logs/r16_enhanced_finalize.log}
INTERVAL=${INTERVAL:-60}

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
exec 9>"$FARO_ROOT/logs/.r16_finalize.lock"
flock -n 9 || exit 0

if [[ -z "$EVAL" ]]; then
  for candidate in \
    "$FARO_ROOT/evals/r16_depth_local_rarity_free_holdout512_diagnostic" \
    "$FARO_ROOT/evals/r16_depth_local_rarity_free_holdout512_enhanced"; do
    if [[ -f "$candidate" ]]; then
      EVAL=$candidate
      break
    fi
  done
fi

while [[ -z "$EVAL" || ! -f "$EVAL" ]]; do
  printf '%s waiting_for_eval=diagnostic_or_enhanced\n' "$(date -Is)" >> "$LOG"
  EVAL=""
  for candidate in \
    "$FARO_ROOT/evals/r16_depth_local_rarity_free_holdout512_diagnostic" \
    "$FARO_ROOT/evals/r16_depth_local_rarity_free_holdout512_enhanced"; do
    if [[ -f "$candidate" ]]; then
      EVAL=$candidate
      break
    fi
  done
  [[ -n "$EVAL" ]] || sleep "$INTERVAL"
done

python3 - "$EVAL" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
rows = payload.get("refseg_overall", {}).get("records", [])
if len(rows) != 512:
    raise SystemExit(f"R16 enhanced eval requires 512 rows, got {len(rows)}")
if any("boundary_iou" not in row for row in rows):
    raise SystemExit("R16 enhanced eval is missing boundary_iou")
PY

[[ -f "$BASE" ]] || { echo "Missing R18 baseline: $BASE" >> "$LOG"; exit 2; }
PYTHONPATH="$FARO_ROOT" python3 "$FARO_ROOT/tools/analyze_selective_eval.py" \
  "$BASE" "$EVAL" --repeats 20000 --output "$OUT" >> "$LOG" 2>&1
PYTHONPATH="$FARO_ROOT" python3 "$FARO_ROOT/tools/analyze_selective_slices.py" \
  "$BASE" "$EVAL" --repeats 20000 --output "$SLICES" >> "$LOG" 2>&1
printf '%s finalized eval=%s analysis=%s slices=%s\n' "$(date -Is)" "$EVAL" "$OUT" "$SLICES" >> "$LOG"
