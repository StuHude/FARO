#!/usr/bin/env bash
set -euo pipefail

FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BASE=${BASE:-$FARO_ROOT/evals/r18_100_confirmation_holdout512}
SFT=${SFT:-$FARO_ROOT/evals/r18_matched_sft_holdout512}
PV=${PV:-$FARO_ROOT/evals/paired_view_holdout512}
OUT=${OUT:-$FARO_ROOT/evals}
mkdir -p "$OUT"

for path in "$BASE" "$SFT"; do
  [[ -s "$path" ]] || { echo "waiting for $path"; exit 3; }
done

run_analysis() {
  local name=$1 candidate=$2
  PYTHONPATH="$FARO_ROOT" python3 "$FARO_ROOT/tools/analyze_selective_eval.py" \
    "$BASE" "$candidate" --section refseg_overall --repeats 20000 --seed 42 \
    --noninferiority-margin 0.01 --min-utility-ci-lower 0.0 \
    --require-utility-ci-positive --min-positive-ciou-ci-lower -0.01 \
    --min-negative-ci-lower -0.01 --output "$OUT/${name}_vs_r18_bootstrap20k.json"
  PYTHONPATH="$FARO_ROOT" python3 "$FARO_ROOT/tools/analyze_selective_slices.py" \
    "$BASE" "$candidate" --section refseg_overall --repeats 20000 --seed 42 \
    --max-drop 0.01 --min-noninferior-slices 2 \
    --output "$OUT/${name}_vs_r18_slices20k.json"
}

if [[ ! -s "$OUT/r18_matched_sft_vs_r18_bootstrap20k.json" ]]; then
  run_analysis r18_matched_sft "$SFT"
fi

if [[ ! -s "$PV" ]]; then
  echo "waiting for $PV"
  exit 3
fi

if [[ -s "$PV" && ! -s "$OUT/paired_view_vs_r18_bootstrap20k.json" ]]; then
  run_analysis paired_view "$PV"
fi

if [[ -s "$PV" && ! -s "$OUT/paired_view_vs_matched_sft_bootstrap20k.json" ]]; then
  PYTHONPATH="$FARO_ROOT" python3 "$FARO_ROOT/tools/analyze_selective_eval.py" \
    "$SFT" "$PV" --section refseg_overall --repeats 20000 --seed 43 \
    --noninferiority-margin 0.01 --min-utility-ci-lower 0.0 \
    --require-utility-ci-positive --min-positive-ciou-ci-lower -0.01 \
    --min-negative-ci-lower -0.01 \
    --output "$OUT/paired_view_vs_matched_sft_bootstrap20k.json"
fi

echo "finalize analysis complete"
