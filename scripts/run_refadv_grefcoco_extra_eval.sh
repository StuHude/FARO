#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export TORCHDYNAMO_DISABLE=1

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python

CONFIG=${1:?config path required}
ADAPTER=${2:?adapter path required}
OUT_DIR=${3:?output dir required}

GREF_ROOT=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/grefcoco
GREF_SCHEMA=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/grefcoco_val.jsonl
COCO_IMG=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014
REFADV_PARQUET=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/ref_adv_s/data/train-00000-of-00001.parquet

mkdir -p "$OUT_DIR"

if [[ ! -f "$GREF_SCHEMA" ]]; then
  "$PY" -m projects.pixvl_idea1.scripts.prepare_grefcoco_schema \
    --grefs-json "$GREF_ROOT/grefs(unc).json" \
    --instances-json "$GREF_ROOT/instances.json" \
    --image-root "$COCO_IMG" \
    --split val \
    --output "$GREF_SCHEMA"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -m projects.pixvl_idea1.eval.eval_refseg \
  --config "$CONFIG" \
  --adapter-path "$ADAPTER" \
  --schema-file "$GREF_SCHEMA" \
  --output "$OUT_DIR/grefcoco_val.json"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -m projects.pixvl_idea1.eval.eval_ref_adv_s \
  --config "$CONFIG" \
  --adapter-path "$ADAPTER" \
  --parquet-file "$REFADV_PARQUET" \
  --output "$OUT_DIR/ref_adv_s.json"

"$PY" - <<PY
import json
from pathlib import Path
out = Path("$OUT_DIR")
gref = json.load(open(out / "grefcoco_val.json", "r", encoding="utf-8"))
refadv = json.load(open(out / "ref_adv_s.json", "r", encoding="utf-8"))
summary = {
    "grefcoco_val_ciou": gref["mean_ciou"],
    "grefcoco_val_num_samples": gref["num_samples"],
    "ref_adv_s_mean_bbox_iou": refadv["mean_bbox_iou"],
    "ref_adv_s_acc50": refadv["acc50"],
    "ref_adv_s_acc75": refadv["acc75"],
    "ref_adv_s_acc90": refadv["acc90"],
    "ref_adv_s_num_samples": refadv["num_samples"],
}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(out / "summary.json")
PY
