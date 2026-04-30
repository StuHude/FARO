#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export PIXVL_TEXT_SIM_DEVICE=cpu
export PIXVL_TEXT_SIM_LOCAL_ONLY=1
export TORCHDYNAMO_DISABLE=1

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
MODEL=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok
ROOT_OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_ckpt1000_eval
mkdir -p "$ROOT_OUT"

CFG_UNIFIED=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_2gpu_unified_opd_rl.py
CFG_ROUTED_RL=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_3gpu_routed_rl.py
CFG_ROUTED_OPD=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_3gpu_routed_opd_rl.py

ADAPTER_UNIFIED=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_2gpu_unified_opd_rl/checkpoint-step-1000/adapter
ADAPTER_ROUTED_RL=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_rl/checkpoint-step-1000/adapter
ADAPTER_ROUTED_OPD=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_opd_rl/checkpoint-step-1000/adapter

SUBSET_ROOT=/mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_500
REFCOCO_DATA=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json
REFCOCO_IMG=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014

run_refcoco_eval() {
  local name="$1"
  local gpu0="$2"
  local gpu1="$3"
  local adapter="$4"
  local temp_dir="${ROOT_OUT}/refcoco_temp_${name}"
  local log_dir="${ROOT_OUT}/refcoco_${name}"
  rm -rf "$temp_dir"
  mkdir -p "$temp_dir" "$log_dir"

  CUDA_VISIBLE_DEVICES="$gpu0" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --model_path "$MODEL" \
    --adapter_path "$adapter" \
    --vq_sam2_path "$MODEL/mask_tokenizer_256x2.pth" \
    --sam2_path "$MODEL/sam2.1_hiera_large.pt" \
    --dataset "$REFCOCO_DATA" \
    --image_folder "$REFCOCO_IMG" \
    --temp_save_dir "$temp_dir" \
    --task_id 0 --num_tasks 2 > "${log_dir}/shard0.log" 2>&1 &
  local p0=$!
  CUDA_VISIBLE_DEVICES="$gpu1" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --model_path "$MODEL" \
    --adapter_path "$adapter" \
    --vq_sam2_path "$MODEL/mask_tokenizer_256x2.pth" \
    --sam2_path "$MODEL/sam2.1_hiera_large.pt" \
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
    --vq_sam2_path "$MODEL/mask_tokenizer_256x2.pth" \
    --sam2_path "$MODEL/sam2.1_hiera_large.pt" \
    --metric-only --quiet-metric \
    --temp_save_dir "$temp_dir" > "${log_dir}/metric.log" 2>&1
}

run_bundle() {
  local name="$1"
  local gpu="$2"
  local config="$3"
  local adapter="$4"
  local out_dir="${ROOT_OUT}/bundle_${name}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m projects.pixvl_idea3.eval.eval_mvp_bundle \
    --config "$config" \
    --adapter-path "$adapter" \
    --relation-schema "${SUBSET_ROOT}/relation_500.jsonl" \
    --geometry-schema "${SUBSET_ROOT}/geometry_500.jsonl" \
    --semantic-schema "${SUBSET_ROOT}/semantic_500.jsonl" \
    --refseg-overall-schema "${SUBSET_ROOT}/refseg_val_500.jsonl" \
    --dlc-schema "${SUBSET_ROOT}/dlc_eval_100.jsonl" \
    --output "${out_dir}/results.json" > "${out_dir}/run.log" 2>&1
}

run_refcoco_eval unified 0 1 "$ADAPTER_UNIFIED" &
P_UNI=$!
run_refcoco_eval routed_rl 2 3 "$ADAPTER_ROUTED_RL" &
P_RRL=$!
run_refcoco_eval routed_opd_rl 4 5 "$ADAPTER_ROUTED_OPD" &
P_ROPD=$!

run_bundle unified 6 "$CFG_UNIFIED" "$ADAPTER_UNIFIED" &
P_BU=$!
run_bundle routed_rl 7 "$CFG_ROUTED_RL" "$ADAPTER_ROUTED_RL" &
P_BR=$!

wait "$P_UNI"
wait "$P_RRL"
wait "$P_ROPD"
wait "$P_BU"
wait "$P_BR"

run_bundle routed_opd_rl 6 "$CFG_ROUTED_OPD" "$ADAPTER_ROUTED_OPD"

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_ckpt1000_eval")

def parse_metric(path: Path):
    text = path.read_text(encoding="utf-8")
    line = [x.strip() for x in text.splitlines() if "REC AP_50:" in x][-1]
    ap50 = float(line.split("REC AP_50:")[1].split("|")[0].strip())
    ciou = float(line.split("RES CIoU:")[1].strip())
    return ap50, ciou

summary = {}
for name in ["unified", "routed_rl", "routed_opd_rl"]:
    ap50, ciou = parse_metric(root / f"refcoco_{name}" / "metric.log")
    bundle = json.load(open(root / f"bundle_{name}" / "results.json", "r", encoding="utf-8"))
    summary[name] = {
        "refcoco_val_ap50": ap50,
        "refcoco_val_ciou": ciou,
        "relation_ciou": bundle["relation"]["mean_ciou"],
        "geometry_ciou": bundle["geometry"]["mean_ciou"],
        "semantic_reward": bundle["semantic"]["mean_reward"],
        "overall_refseg_ciou": bundle["refseg_overall"]["mean_ciou"],
        "dlc_reward": bundle["dlc"]["mean_reward"],
    }

out = root / "summary_1000.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
PY
