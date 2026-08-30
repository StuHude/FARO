#!/usr/bin/env bash
set -euo pipefail

# R14: minimal two-GPU depth-local geometry screen. The shared submit wrapper
# enforces dna-* naming, ailab-dnacoding, and every positive tag in rjob_tags.txt.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_depth_local_geometry_10step_2gpu.py}
DATA=${SAMTOK_SELECTIVE_DATA:-$FARO_ROOT/data/fepo_existence/egfepo_train_5120.jsonl}
JOB_NAME=${JOB_NAME:-dna-fepo-depth-local-geometry-10step-2g}

[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
[[ -f "$DATA" ]] || { echo "Missing training data: $DATA" >&2; exit 2; }
rows=$(awk 'NF {n++} END {print n+0}' "$DATA")
(( rows >= 5000 )) || { echo "R14 requires at least 5000 rows, got $rows" >&2; exit 2; }

export CONFIG SAMTOK_SELECTIVE_DATA="$DATA" JOB_NAME
exec bash "$SCRIPT_DIR/submit_samtok_tb_gppo.sh"
