#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export PIXVL_TEXT_SIM_DEVICE=cpu
export PIXVL_TEXT_SIM_LOCAL_ONLY=1
export TORCHDYNAMO_DISABLE=1

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
MODEL=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok
REFCOCO_DATASET=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json
REFCOCO_IMAGE_FOLDER=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014

SUBSET_ROOT=/mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_500
OUT_ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/eval_24_bs3_models

mkdir -p "$OUT_ROOT"
rm -rf "$OUT_ROOT"/*

CFG_BASE=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_2gpu_samtok_bs3_group8x4_1000_unified_opd_rl.py
CFG_UNIFIED=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_2gpu_samtok_bs3_group8x4_1000_unified_opd_rl.py
CFG_ROUTED_RL=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_2gpu_samtok_bs3_group8x4_1000_routed_rl.py
CFG_ROUTED_OPD=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_2gpu_samtok_bs3_group8x4_1000_routed_opd_rl.py

ADAPTER_UNIFIED=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/2gpu_samtok_bs3_group8x4_1000_unified_opd_rl/adapter
ADAPTER_ROUTED_RL=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/2gpu_samtok_bs3_group8x4_1000_routed_rl/adapter
ADAPTER_ROUTED_OPD=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/2gpu_samtok_bs3_group8x4_1000_routed_opd_rl/adapter

run_refcoco_pair() {
  local name="$1"
  local gpu0="$2"
  local gpu1="$3"
  local adapter="${4:-}"
  local temp_dir="$OUT_ROOT/refcoco_temp_${name}"
  local log_dir="$OUT_ROOT/refcoco_${name}"
  mkdir -p "$log_dir"
  rm -rf "$temp_dir"
  mkdir -p "$temp_dir"

  if [[ -n "$adapter" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu0" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path "$MODEL" \
      --adapter_path "$adapter" \
      --vq_sam2_path "$MODEL/mask_tokenizer_256x2.pth" \
      --sam2_path "$MODEL/sam2.1_hiera_large.pt" \
      --dataset "$REFCOCO_DATASET" \
      --image_folder "$REFCOCO_IMAGE_FOLDER" \
      --temp_save_dir "$temp_dir" \
      --task_id 0 --num_tasks 2 > "$log_dir/shard0.log" 2>&1 &
    local p0=$!
    CUDA_VISIBLE_DEVICES="$gpu1" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path "$MODEL" \
      --adapter_path "$adapter" \
      --vq_sam2_path "$MODEL/mask_tokenizer_256x2.pth" \
      --sam2_path "$MODEL/sam2.1_hiera_large.pt" \
      --dataset "$REFCOCO_DATASET" \
      --image_folder "$REFCOCO_IMAGE_FOLDER" \
      --temp_save_dir "$temp_dir" \
      --task_id 1 --num_tasks 2 > "$log_dir/shard1.log" 2>&1 &
    local p1=$!
  else
    CUDA_VISIBLE_DEVICES="$gpu0" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path "$MODEL" \
      --vq_sam2_path "$MODEL/mask_tokenizer_256x2.pth" \
      --sam2_path "$MODEL/sam2.1_hiera_large.pt" \
      --dataset "$REFCOCO_DATASET" \
      --image_folder "$REFCOCO_IMAGE_FOLDER" \
      --temp_save_dir "$temp_dir" \
      --task_id 0 --num_tasks 2 > "$log_dir/shard0.log" 2>&1 &
    local p0=$!
    CUDA_VISIBLE_DEVICES="$gpu1" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path "$MODEL" \
      --vq_sam2_path "$MODEL/mask_tokenizer_256x2.pth" \
      --sam2_path "$MODEL/sam2.1_hiera_large.pt" \
      --dataset "$REFCOCO_DATASET" \
      --image_folder "$REFCOCO_IMAGE_FOLDER" \
      --temp_save_dir "$temp_dir" \
      --task_id 1 --num_tasks 2 > "$log_dir/shard1.log" 2>&1 &
    local p1=$!
  fi

  wait "$p0"
  wait "$p1"

  if [[ -n "$adapter" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu0" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path "$MODEL" \
      --adapter_path "$adapter" \
      --vq_sam2_path "$MODEL/mask_tokenizer_256x2.pth" \
      --sam2_path "$MODEL/sam2.1_hiera_large.pt" \
      --metric-only --quiet-metric \
      --temp_save_dir "$temp_dir" > "$log_dir/metric.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu0" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path "$MODEL" \
      --vq_sam2_path "$MODEL/mask_tokenizer_256x2.pth" \
      --sam2_path "$MODEL/sam2.1_hiera_large.pt" \
      --metric-only --quiet-metric \
      --temp_save_dir "$temp_dir" > "$log_dir/metric.log" 2>&1
  fi
}

run_bundle() {
  local name="$1"
  local gpu="$2"
  local config="$3"
  local adapter="$4"
  local out_dir="$OUT_ROOT/bundle_${name}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.pixvl_idea3.eval.eval_mvp_bundle \
    --config "$config" \
    --adapter-path "$adapter" \
    --relation-schema "$SUBSET_ROOT/relation_500.jsonl" \
    --geometry-schema "$SUBSET_ROOT/geometry_500.jsonl" \
    --semantic-schema "$SUBSET_ROOT/semantic_500.jsonl" \
    --refseg-overall-schema "$SUBSET_ROOT/refseg_val_500.jsonl" \
    --dlc-schema "$SUBSET_ROOT/dlc_eval_100.jsonl" \
    --output "$out_dir/results.json" > "$out_dir/run.log" 2>&1
}

run_dlc() {
  local name="$1"
  local gpu="$2"
  local config="$3"
  local adapter="$4"
  local out_dir="$OUT_ROOT/dlc_${name}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.pixvl_idea1.eval.eval_dlc \
    --config "$config" \
    --adapter-path "$adapter" \
    --schema-file /mnt/pfs/xiaoyicheng/data/pixvl_idea3/schemas/dlc_eval.jsonl \
    --output "$out_dir/results.json" > "$out_dir/run.log" 2>&1
}

# 4-model official RefCOCO full-val
run_refcoco_pair base 0 1 "" &
P_BASE_REF=$!
run_refcoco_pair unified 2 3 "$ADAPTER_UNIFIED" &
P_UNI_REF=$!
run_refcoco_pair routed_rl 4 5 "$ADAPTER_ROUTED_RL" &
P_RRL_REF=$!
run_refcoco_pair routed_opd_rl 6 7 "$ADAPTER_ROUTED_OPD" &
P_ROPD_REF=$!

wait "$P_BASE_REF"
wait "$P_UNI_REF"
wait "$P_RRL_REF"
wait "$P_ROPD_REF"

# 4-model slice/bundle eval
run_bundle base 0 "$CFG_BASE" "" &
P_BASE_BUNDLE=$!
run_bundle unified 1 "$CFG_UNIFIED" "$ADAPTER_UNIFIED" &
P_UNI_BUNDLE=$!
run_bundle routed_rl 2 "$CFG_ROUTED_RL" "$ADAPTER_ROUTED_RL" &
P_RRL_BUNDLE=$!
run_bundle routed_opd_rl 3 "$CFG_ROUTED_OPD" "$ADAPTER_ROUTED_OPD" &
P_ROPD_BUNDLE=$!

# Also compute DLC standalone as a sanity fallback
run_dlc unified 4 "$CFG_UNIFIED" "$ADAPTER_UNIFIED" &
P_UNI_DLC=$!
run_dlc routed_rl 5 "$CFG_ROUTED_RL" "$ADAPTER_ROUTED_RL" &
P_RRL_DLC=$!
run_dlc routed_opd_rl 6 "$CFG_ROUTED_OPD" "$ADAPTER_ROUTED_OPD" &
P_ROPD_DLC=$!
run_dlc base 7 "$CFG_BASE" "" &
P_BASE_DLC=$!

wait "$P_BASE_BUNDLE"
wait "$P_UNI_BUNDLE"
wait "$P_RRL_BUNDLE"
wait "$P_ROPD_BUNDLE"
wait "$P_UNI_DLC"
wait "$P_RRL_DLC"
wait "$P_ROPD_DLC"
wait "$P_BASE_DLC"

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/eval_24_bs3_models")

def parse_refcoco_metric(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if "REC AP_50:" in line]
    line = lines[-1]
    ap50 = float(line.split("REC AP_50:")[1].split("|")[0].strip())
    ciou = float(line.split("RES CIoU:")[1].strip())
    return ap50, ciou

summary = {}
for name in ["base", "unified", "routed_rl", "routed_opd_rl"]:
    ap50, ciou = parse_refcoco_metric(root / f"refcoco_{name}" / "metric.log")
    bundle = json.load(open(root / f"bundle_{name}" / "results.json", "r", encoding="utf-8"))
    dlc = json.load(open(root / f"dlc_{name}" / "results.json", "r", encoding="utf-8"))
    summary[name] = {
        "refcoco_val_ap50": ap50,
        "refcoco_val_ciou": ciou,
        "relation_ciou": bundle["relation"]["mean_ciou"],
        "geometry_ciou": bundle["geometry"]["mean_ciou"],
        "semantic_reward": bundle["semantic"]["mean_reward"],
        "overall_refseg_ciou": bundle["refseg_overall"]["mean_ciou"],
        "dlc_reward": dlc["mean_reward"],
    }

out = root / "summary_24.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
PY
