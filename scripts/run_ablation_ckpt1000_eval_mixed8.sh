#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export PIXVL_TEXT_SIM_DEVICE=cpu
export PIXVL_TEXT_SIM_LOCAL_ONLY=1
export TORCHDYNAMO_DISABLE=1

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
MODEL=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok
VQ_SAM2_PATH=$MODEL/mask_tokenizer_256x2.pth
SAM2_PATH=$MODEL/sam2.1_hiera_large.pt
REFCOCO_DATA=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json
REFCOCO_IMG=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014
SUBSET_ROOT=/mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_2000
DLC_JSON=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/DLC-bench.json
DLC_IMG=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/images
JUDGE_SCRIPT=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/eval_dlc_with_local_judge.py
OUT_ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8

CFG_NO_BUCKET=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_3gpu_routed_opd_rl_no_bucket_opd.py
CFG_SHUFFLED=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_2gpu_routed_rl_shuffled_labels.py

ADAPTER_NO_BUCKET=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_opd_rl_no_bucket_opd/checkpoint-step-1000/adapter
ADAPTER_SHUFFLED=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_2gpu_routed_rl_shuffled_labels/checkpoint-step-1000/adapter

mkdir -p "$OUT_ROOT"

run_refcoco_pair() {
  local name="$1"
  local gpu0="$2"
  local gpu1="$3"
  local adapter="$4"
  local temp_dir="$OUT_ROOT/refcoco_temp_${name}"
  local log_dir="$OUT_ROOT/refcoco_${name}"
  mkdir -p "$temp_dir" "$log_dir"

  CUDA_VISIBLE_DEVICES="$gpu0" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --model_path "$MODEL" \
    --adapter_path "$adapter" \
    --vq_sam2_path "$VQ_SAM2_PATH" \
    --sam2_path "$SAM2_PATH" \
    --dataset "$REFCOCO_DATA" \
    --image_folder "$REFCOCO_IMG" \
    --temp_save_dir "$temp_dir" \
    --task_id 0 --num_tasks 2 > "${log_dir}/shard0.log" 2>&1 &
  local p0=$!
  CUDA_VISIBLE_DEVICES="$gpu1" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --model_path "$MODEL" \
    --adapter_path "$adapter" \
    --vq_sam2_path "$VQ_SAM2_PATH" \
    --sam2_path "$SAM2_PATH" \
    --dataset "$REFCOCO_DATA" \
    --image_folder "$REFCOCO_IMG" \
    --temp_save_dir "$temp_dir" \
    --task_id 1 --num_tasks 2 > "${log_dir}/shard1.log" 2>&1 &
  local p1=$!
  wait "$p0"
  wait "$p1"
  CUDA_VISIBLE_DEVICES="$gpu0" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --model_path "$MODEL" \
    --adapter_path "$adapter" \
    --vq_sam2_path "$VQ_SAM2_PATH" \
    --sam2_path "$SAM2_PATH" \
    --metric-only --quiet-metric \
    --temp_save_dir "$temp_dir" > "${log_dir}/metric.log" 2>&1
}

run_dlc_official_full() {
  local name="$1"
  local gpu="$2"
  local adapter="$3"
  local out_dir="$OUT_ROOT/dlc_official_${name}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_dam_infer \
    --model_path "$MODEL" \
    --adapter_path "$adapter" \
    --vq_sam2_path "$VQ_SAM2_PATH" \
    --sam2_path "$SAM2_PATH" \
    --dataset "$DLC_JSON" \
    --image_root "$DLC_IMG" \
    --output_path "$out_dir/raw.json" > "$out_dir/generate.log" 2>&1

  "$PY" - <<PY
import json
from pathlib import Path
raw = json.load(open("$out_dir/raw.json", "r", encoding="utf-8"))
pred = {}
for item in raw:
    for sample in item["mask_samples"]:
        pred[str(sample["ann_id"])] = sample["pred_caption"]
Path("$out_dir/pred.json").write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
PY

  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$JUDGE_SCRIPT" \
    --pred "$out_dir/pred.json" \
    --output "$out_dir/eval.json" \
    --device cuda:0 > "$out_dir/judge.log" 2>&1
}

run_split_queue() {
  local name="$1"
  local gpu="$2"
  local config="$3"
  local adapter="$4"
  local out_dir="$OUT_ROOT/split_${name}"
  mkdir -p "$out_dir"

  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.pixvl_idea1.eval.eval_refseg \
    --config "$config" --adapter-path "$adapter" \
    --schema-file "$SUBSET_ROOT/relation_2000.jsonl" \
    --output "$out_dir/relation.json"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.pixvl_idea1.eval.eval_refseg \
    --config "$config" --adapter-path "$adapter" \
    --schema-file "$SUBSET_ROOT/geometry_2000.jsonl" \
    --output "$out_dir/geometry.json"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.pixvl_idea1.eval.eval_dlc \
    --config "$config" --adapter-path "$adapter" \
    --schema-file "$SUBSET_ROOT/semantic_2000.jsonl" \
    --output "$out_dir/semantic.json"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.pixvl_idea1.eval.eval_refseg \
    --config "$config" --adapter-path "$adapter" \
    --schema-file "$SUBSET_ROOT/refseg_val_2000.jsonl" \
    --output "$out_dir/overall.json"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.pixvl_idea1.eval.eval_dlc \
    --config "$config" --adapter-path "$adapter" \
    --schema-file "$SUBSET_ROOT/dlc_eval_100.jsonl" \
    --output "$out_dir/dlc_reward.json"
}

# 8-GPU mixed scheduling:
# 0-3 RefCOCO, 4-5 official DLC, 6-7 split-eval queues.
run_refcoco_pair no_bucket 0 1 "$ADAPTER_NO_BUCKET" &
P0=$!
run_refcoco_pair shuffled 2 3 "$ADAPTER_SHUFFLED" &
P1=$!
run_dlc_official_full no_bucket 4 "$ADAPTER_NO_BUCKET" &
P2=$!
run_dlc_official_full shuffled 5 "$ADAPTER_SHUFFLED" &
P3=$!
run_split_queue no_bucket 6 "$CFG_NO_BUCKET" "$ADAPTER_NO_BUCKET" &
P4=$!
run_split_queue shuffled 7 "$CFG_SHUFFLED" "$ADAPTER_SHUFFLED" &
P5=$!

wait "$P0" "$P1" "$P2" "$P3" "$P4" "$P5"

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8")

def parse_metric(path: Path):
    text = path.read_text(encoding="utf-8")
    line = [x.strip() for x in text.splitlines() if "REC AP_50:" in x][-1]
    ap50 = float(line.split("REC AP_50:")[1].split("|")[0].strip())
    ciou = float(line.split("RES CIoU:")[1].strip())
    return ap50, ciou

summary = {}
for name in ["no_bucket", "shuffled"]:
    ap50, ciou = parse_metric(root / f"refcoco_{name}" / "metric.log")
    split_root = root / f"split_{name}"
    relation = json.load(open(split_root / "relation.json", "r", encoding="utf-8"))
    geometry = json.load(open(split_root / "geometry.json", "r", encoding="utf-8"))
    semantic = json.load(open(split_root / "semantic.json", "r", encoding="utf-8"))
    overall = json.load(open(split_root / "overall.json", "r", encoding="utf-8"))
    dlc_reward = json.load(open(split_root / "dlc_reward.json", "r", encoding="utf-8"))
    dlc_official = json.load(open(root / f"dlc_official_{name}" / "eval.json", "r", encoding="utf-8"))
    summary[name] = {
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
    }

out = root / "summary_ckpt1000.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
PY
