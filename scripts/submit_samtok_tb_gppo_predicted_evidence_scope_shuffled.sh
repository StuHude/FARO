#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_shuffled_10step_2gpu.py}
DATA=${SAMTOK_SELECTIVE_DATA:-$FARO_ROOT/data/fepo_existence/egfepo_train_5120.jsonl}
JOB_NAME=${JOB_NAME:-dna-fepo-predicted-evidence-shuffled-10step-2g}
[[ -f "$CONFIG" && -f "$DATA" ]] || { echo "Missing PES shuffled input" >&2; exit 2; }
rows=$(awk 'NF {n++} END {print n+0}' "$DATA")
(( rows >= 5000 )) || { echo "PES shuffled requires at least 5000 rows" >&2; exit 2; }
case "$JOB_NAME" in dna-*) ;; *) echo "JOB_NAME must start with dna-" >&2; exit 2 ;; esac
export CONFIG SAMTOK_SELECTIVE_DATA="$DATA" JOB_NAME
exec bash "$SCRIPT_DIR/submit_samtok_tb_gppo.sh"
