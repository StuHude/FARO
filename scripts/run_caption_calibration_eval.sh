#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export PIXVL_TEXT_SIM_DEVICE=cpu
export PIXVL_TEXT_SIM_LOCAL_ONLY=1
export TORCHDYNAMO_DISABLE=1

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
MODEL=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok
ADAPTER=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_sft_8gpu/adapter
CONFIG=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_caption_calibration_sft_8gpu.py
VQ_SAM2_PATH=$MODEL/mask_tokenizer_256x2.pth
SAM2_PATH=$MODEL/sam2.1_hiera_large.pt

OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_eval
REFCOCO_DATA=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json
REFCOCO_IMG=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014
SUBSET_ROOT=/mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_2000
DLC_JSON=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/DLC-bench.json
DLC_IMG=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/images
JUDGE_SCRIPT=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/eval_dlc_with_local_judge.py

mkdir -p "$OUT/refcoco/logs" "$OUT/refcoco/temp" "$OUT/dlc_official/shards" "$OUT/split" "$OUT/extra_benches"

# Phase 1: official RefCOCO on 8 GPUs
for task in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$task" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --model_path "$MODEL" \
    --adapter_path "$ADAPTER" \
    --vq_sam2_path "$VQ_SAM2_PATH" \
    --sam2_path "$SAM2_PATH" \
    --dataset "$REFCOCO_DATA" \
    --image_folder "$REFCOCO_IMG" \
    --temp_save_dir "$OUT/refcoco/temp" \
    --task_id "$task" --num_tasks 8 > "$OUT/refcoco/logs/shard${task}.log" 2>&1 &
done
wait

CUDA_VISIBLE_DEVICES=0 "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
  --model_path "$MODEL" \
  --adapter_path "$ADAPTER" \
  --vq_sam2_path "$VQ_SAM2_PATH" \
  --sam2_path "$SAM2_PATH" \
  --metric-only --quiet-metric \
  --temp_save_dir "$OUT/refcoco/temp" > "$OUT/refcoco/metric.log" 2>&1

# Phase 2: official DLC generation on 8 GPUs, then judge
for task in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$task" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_dam_infer \
    --model_path "$MODEL" \
    --adapter_path "$ADAPTER" \
    --vq_sam2_path "$VQ_SAM2_PATH" \
    --sam2_path "$SAM2_PATH" \
    --dataset "$DLC_JSON" \
    --image_root "$DLC_IMG" \
    --task_id "$task" --num_tasks 8 \
    --output_path "$OUT/dlc_official/shards/raw_${task}.json" > "$OUT/dlc_official/shards/log_${task}.log" 2>&1 &
done
wait

"$PY" - <<PY
import json
from pathlib import Path
root = Path("$OUT/dlc_official")
merged = []
pred = {}
for shard in sorted((root / "shards").glob("raw_*.json")):
    data = json.load(open(shard, "r", encoding="utf-8"))
    merged.extend(data)
    for item in data:
        for sample in item["mask_samples"]:
            pred[str(sample["ann_id"])] = sample["pred_caption"]
(root / "raw.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
(root / "pred.json").write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
PY

CUDA_VISIBLE_DEVICES=0 "$PY" "$JUDGE_SCRIPT" \
  --pred "$OUT/dlc_official/pred.json" \
  --output "$OUT/dlc_official/eval.json" \
  --device cuda:0 > "$OUT/dlc_official/judge.log" 2>&1

# Phase 3: split eval + extra benches
CUDA_VISIBLE_DEVICES=0 "$PY" -m projects.pixvl_idea1.eval.eval_refseg \
  --config "$CONFIG" \
  --adapter-path "$ADAPTER" \
  --schema-file "$SUBSET_ROOT/relation_2000.jsonl" \
  --output "$OUT/split/relation.json" &
P0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -m projects.pixvl_idea1.eval.eval_refseg \
  --config "$CONFIG" \
  --adapter-path "$ADAPTER" \
  --schema-file "$SUBSET_ROOT/geometry_2000.jsonl" \
  --output "$OUT/split/geometry.json" &
P1=$!
CUDA_VISIBLE_DEVICES=2 "$PY" -m projects.pixvl_idea1.eval.eval_dlc \
  --config "$CONFIG" \
  --adapter-path "$ADAPTER" \
  --schema-file "$SUBSET_ROOT/semantic_2000.jsonl" \
  --output "$OUT/split/semantic.json" &
P2=$!
CUDA_VISIBLE_DEVICES=3 "$PY" -m projects.pixvl_idea1.eval.eval_refseg \
  --config "$CONFIG" \
  --adapter-path "$ADAPTER" \
  --schema-file "$SUBSET_ROOT/refseg_val_2000.jsonl" \
  --output "$OUT/split/overall.json" &
P3=$!
CUDA_VISIBLE_DEVICES=4 "$PY" -m projects.pixvl_idea1.eval.eval_dlc \
  --config "$CONFIG" \
  --adapter-path "$ADAPTER" \
  --schema-file "$SUBSET_ROOT/dlc_eval_100.jsonl" \
  --output "$OUT/split/dlc_reward.json" &
P4=$!
CUDA_VISIBLE_DEVICES=5 bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_refadv_grefcoco_extra_eval.sh \
  "$CONFIG" "$ADAPTER" "$OUT/extra_benches" > "$OUT/extra_benches/run.log" 2>&1 &
P5=$!
wait "$P0" "$P1" "$P2" "$P3" "$P4" "$P5"

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_eval")

def parse_metric(path: Path):
    text = path.read_text(encoding="utf-8")
    line = [x.strip() for x in text.splitlines() if "REC AP_50:" in x][-1]
    ap50 = float(line.split("REC AP_50:")[1].split("|")[0].strip())
    ciou = float(line.split("RES CIoU:")[1].strip())
    return ap50, ciou

ap50, ciou = parse_metric(root / "refcoco" / "metric.log")
relation = json.load(open(root / "split" / "relation.json", "r", encoding="utf-8"))
geometry = json.load(open(root / "split" / "geometry.json", "r", encoding="utf-8"))
semantic = json.load(open(root / "split" / "semantic.json", "r", encoding="utf-8"))
overall = json.load(open(root / "split" / "overall.json", "r", encoding="utf-8"))
dlc_reward = json.load(open(root / "split" / "dlc_reward.json", "r", encoding="utf-8"))
dlc_official = json.load(open(root / "dlc_official" / "eval.json", "r", encoding="utf-8"))
extra = json.load(open(root / "extra_benches" / "summary.json", "r", encoding="utf-8"))

summary = {
    "refcoco_val_ap50": ap50,
    "refcoco_val_ciou": ciou,
    "relation_ciou": relation["mean_ciou"],
    "geometry_ciou": geometry["mean_ciou"],
    "semantic_reward": semantic["mean_reward"],
    "overall_refseg_ciou": overall["mean_ciou"],
    "dlc_reward": dlc_reward["mean_reward"],
    "dlc_official_avg_pos": dlc_official["avg_pos"],
    "dlc_official_avg_neg": dlc_official["avg_neg"],
    "dlc_official_avg": dlc_official["avg"],
    "extra_benches": extra["extra_benches"],
}

(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(root / "summary.json")
PY
