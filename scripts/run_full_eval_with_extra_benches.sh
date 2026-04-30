#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?config path required}
ADAPTER=${2:?adapter path required}
OUT_DIR=${3:?output dir required}

mkdir -p "$OUT_DIR"

bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_refadv_grefcoco_extra_eval.sh \
  "$CONFIG" \
  "$ADAPTER" \
  "$OUT_DIR/extra_benches"

python - <<PY
import json
from pathlib import Path

root = Path("$OUT_DIR")
extra = json.load(open(root / "extra_benches" / "summary.json", "r", encoding="utf-8"))
summary = {"extra_benches": extra}
(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(root / "summary.json")
PY
