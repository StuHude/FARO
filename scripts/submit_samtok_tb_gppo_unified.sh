#!/usr/bin/env bash
set -euo pipefail

# R4 uses the single-sentinel contract and the mandatory >=5k-row training set.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_tail_balanced_unified_sentinel_10step_2gpu.py}
DATA=${SAMTOK_SELECTIVE_DATA:-$FARO_ROOT/data/fepo_existence/egfepo_train_5120.jsonl}
export CONFIG SAMTOK_SELECTIVE_DATA="$DATA"

[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
[[ -f "$DATA" ]] || { echo "Missing training data: $DATA" >&2; exit 2; }
rows=$(awk 'NF {n++} END {print n+0}' "$DATA")
(( rows >= 5000 )) || { echo "R4 requires at least 5000 rows, got $rows" >&2; exit 2; }

JOB_NAME=${JOB_NAME:-dna-samtok-fepo-unified-sentinel-10step-2g}
case "$JOB_NAME" in dna-*) ;; *) echo "JOB_NAME must start with dna-" >&2; exit 2 ;; esac
export JOB_NAME

exec bash "$SCRIPT_DIR/submit_samtok_tb_gppo.sh"
