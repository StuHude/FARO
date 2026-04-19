#!/usr/bin/env bash

set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PIXVL_TEXT_SIM_DEVICE="${PIXVL_TEXT_SIM_DEVICE:-cpu}"
export PIXVL_TEXT_SIM_LOCAL_ONLY="${PIXVL_TEXT_SIM_LOCAL_ONLY:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
ACC=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/accelerate
ROOT_OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3
LOG_DIR=${ROOT_OUT}/run_2gpu_bs4_2000_logs
mkdir -p "$LOG_DIR"

CFG_UNIFIED=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_2gpu_bs4_2000_unified_opd_rl.py
CFG_ROUTED_RL=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_2gpu_bs4_2000_routed_rl.py
CFG_ROUTED_OPD=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_2gpu_bs4_2000_routed_opd_rl.py

OUT_UNIFIED=${ROOT_OUT}/2gpu_bs4_2000_unified_opd_rl
OUT_ROUTED_RL=${ROOT_OUT}/2gpu_bs4_2000_routed_rl
OUT_ROUTED_OPD=${ROOT_OUT}/2gpu_bs4_2000_routed_opd_rl

echo "=== TRAIN START $(date '+%F %T') ==="

CUDA_VISIBLE_DEVICES=0,1 "$ACC" launch \
  --main_process_port 30020 \
  --num_processes 2 \
  --mixed_precision bf16 \
  -m projects.pixvl_idea1.trainers.joint_opd_rl_trainer \
  --config "$CFG_UNIFIED" >> "${LOG_DIR}/train_unified.log" 2>&1 &
PID_UNIFIED=$!

CUDA_VISIBLE_DEVICES=2,3 "$ACC" launch \
  --main_process_port 30022 \
  --num_processes 2 \
  --mixed_precision bf16 \
  -m projects.pixvl_idea3.trainers.joint_routed_opd_rl_trainer \
  --config "$CFG_ROUTED_RL" >> "${LOG_DIR}/train_routed_rl.log" 2>&1 &
PID_ROUTED_RL=$!

CUDA_VISIBLE_DEVICES=4,5 "$ACC" launch \
  --main_process_port 30024 \
  --num_processes 2 \
  --mixed_precision bf16 \
  -m projects.pixvl_idea3.trainers.joint_routed_opd_rl_trainer \
  --config "$CFG_ROUTED_OPD" >> "${LOG_DIR}/train_routed_opd.log" 2>&1 &
PID_ROUTED_OPD=$!

wait "$PID_UNIFIED"
wait "$PID_ROUTED_RL"
wait "$PID_ROUTED_OPD"

echo "=== TRAIN DONE $(date '+%F %T') ==="

# Official RefCOCO full-val evaluation for base + 3 trained ckpts
mkdir -p ${ROOT_OUT}/formal_official_refcoco
run_refcoco_eval() {
  local name="$1"
  local gpu0="$2"
  local gpu1="$3"
  local adapter="$4"
  local temp_dir="/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/temp_save/refcoco_${name}_2000"
  local log_dir="${ROOT_OUT}/formal_official_refcoco/${name}"
  mkdir -p "$log_dir"
  rm -rf "$temp_dir"
  mkdir -p "$temp_dir"
  if [ -n "$adapter" ]; then
    CUDA_VISIBLE_DEVICES=$gpu0 "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co \
      --adapter_path "$adapter" \
      --vq_sam2_path /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/mask_tokenizer_256x2.pth \
      --sam2_path /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/sam2.1_hiera_large.pt \
      --dataset /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json \
      --image_folder /mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014 \
      --temp_save_dir "$temp_dir" \
      --task_id 0 --num_tasks 2 >> "${log_dir}/shard0.log" 2>&1 &
    P0=$!
    CUDA_VISIBLE_DEVICES=$gpu1 "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co \
      --adapter_path "$adapter" \
      --vq_sam2_path /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/mask_tokenizer_256x2.pth \
      --sam2_path /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/sam2.1_hiera_large.pt \
      --dataset /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json \
      --image_folder /mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014 \
      --temp_save_dir "$temp_dir" \
      --task_id 1 --num_tasks 2 >> "${log_dir}/shard1.log" 2>&1 &
    P1=$!
  else
    CUDA_VISIBLE_DEVICES=$gpu0 "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co \
      --vq_sam2_path /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/mask_tokenizer_256x2.pth \
      --sam2_path /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/sam2.1_hiera_large.pt \
      --dataset /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json \
      --image_folder /mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014 \
      --temp_save_dir "$temp_dir" \
      --task_id 0 --num_tasks 2 >> "${log_dir}/shard0.log" 2>&1 &
    P0=$!
    CUDA_VISIBLE_DEVICES=$gpu1 "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
      --model_path /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co \
      --vq_sam2_path /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/mask_tokenizer_256x2.pth \
      --sam2_path /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/sam2.1_hiera_large.pt \
      --dataset /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json \
      --image_folder /mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014 \
      --temp_save_dir "$temp_dir" \
      --task_id 1 --num_tasks 2 >> "${log_dir}/shard1.log" 2>&1 &
    P1=$!
  fi
  local target
  target=$(wc -l < /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json)
  while true; do
    local count
    count=$(find "$temp_dir" -maxdepth 1 -type f | wc -l)
    echo "[$name] $(date '+%F %T') ${count}/${target}"
    if [ "$count" -ge "$target" ]; then
      break
    fi
    sleep 180
  done
  wait "$P0"; wait "$P1"
  CUDA_VISIBLE_DEVICES=$gpu0 "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --metric-only --quiet-metric --temp_save_dir "$temp_dir" > "${log_dir}/metric.log" 2>&1
}

run_refcoco_eval base 0 1 ""
run_refcoco_eval unified 2 3 "${OUT_UNIFIED}/adapter"
run_refcoco_eval routed_rl 4 5 "${OUT_ROUTED_RL}/adapter"
run_refcoco_eval routed_opd_rl 6 7 "${OUT_ROUTED_OPD}/adapter"

# Larger-slice MVP eval using fixed formal subsets
python - <<'PY'
import json, hashlib
from pathlib import Path
root = Path('/mnt/pfs/xiaoyicheng/data/pixvl_idea3/schemas')
out = Path('/mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_500')
out.mkdir(parents=True, exist_ok=True)
configs = [
 ('semantic_slice_eval.jsonl', 'semantic_500.jsonl', 500),
 ('relation_slice_eval.jsonl', 'relation_500.jsonl', 500),
 ('geometry_slice_eval.jsonl', 'geometry_500.jsonl', 500),
 ('refseg_val_routed.jsonl', 'refseg_val_500.jsonl', 500),
 ('dlc_eval.jsonl', 'dlc_eval_100.jsonl', 100),
]
for src_name, dst_name, n in configs:
    rows=[]
    with (root/src_name).open('r',encoding='utf-8') as f:
        for line in f:
            obj=json.loads(line)
            h=int(hashlib.sha1(obj['id'].encode()).hexdigest()[:12],16)
            rows.append((h, line))
    rows.sort(key=lambda x: x[0])
    picked=[line for _,line in rows[:min(n,len(rows))]]
    (out/dst_name).write_text(''.join(picked), encoding='utf-8')
PY

run_bundle() {
  local name="$1"
  local gpu="$2"
  local config="$3"
  local adapter="$4"
  local output="${ROOT_OUT}/mvp_bundle_${name}_formal/results.json"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -m projects.pixvl_idea3.eval.eval_mvp_bundle \
    --config "$config" \
    --adapter-path "$adapter" \
    --relation-schema /mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_500/relation_500.jsonl \
    --geometry-schema /mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_500/geometry_500.jsonl \
    --semantic-schema /mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_500/semantic_500.jsonl \
    --refseg-overall-schema /mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_500/refseg_val_500.jsonl \
    --dlc-schema /mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_500/dlc_eval_100.jsonl \
    --output "$output" >> "${LOG_DIR}/bundle_${name}.log" 2>&1
}

run_bundle base 0 /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_routed_opd_rl.py /mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage1_joint_sft_maxmem/checkpoint-step-800/adapter
run_bundle unified 0 "$CFG_UNIFIED" "${OUT_UNIFIED}/adapter"
run_bundle routed_rl 0 "$CFG_ROUTED_RL" "${OUT_ROUTED_RL}/adapter"
run_bundle routed_opd_rl 0 "$CFG_ROUTED_OPD" "${OUT_ROUTED_OPD}/adapter"

python - <<'PY'
import json
from pathlib import Path

root = Path('/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3')
summary = {}

# base official numbers from previous verified run
summary['base'] = {
    'refcoco_val_ap50': 0.9346322454943985,
    'refcoco_val_ciou': 0.8316177318964235,
}

# parse official refcoco metric logs for trained models
for name in ['unified', 'routed_rl', 'routed_opd_rl']:
    key = name
    metric_log = root / 'formal_official_refcoco' / key / 'metric.log'
    text = metric_log.read_text(encoding='utf-8')
    lines = [line.strip() for line in text.splitlines() if 'REC AP_50:' in line]
    rec_line = lines[-1]
    part1 = rec_line.split('REC AP_50:')[1].split('|')[0].strip()
    part2 = rec_line.split('RES CIoU:')[1].strip()
    summary[name] = {
        'refcoco_val_ap50': float(part1),
        'refcoco_val_ciou': float(part2),
    }

# merge bundle eval
summary['base'].update(json.load(open(root / 'mvp_bundle_base_formal' / 'results.json')))
summary['unified'].update(json.load(open(root / 'mvp_bundle_unified_formal' / 'results.json')))
summary['routed_rl'].update(json.load(open(root / 'mvp_bundle_routed_rl_formal' / 'results.json')))
summary['routed_opd_rl'].update(json.load(open(root / 'mvp_bundle_routed_opd_rl_formal' / 'results.json')))

out = root / 'mvp_24_metrics_formal_summary.json'
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(out)
PY

echo "=== ALL DONE $(date '+%F %T') ==="
