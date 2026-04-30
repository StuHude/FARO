#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export PIXVL_TEXT_SIM_DEVICE=cpu
export PIXVL_TEXT_SIM_LOCAL_ONLY=1
export TORCHDYNAMO_DISABLE=1

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
SUBSET_ROOT=/mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_2000
OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8/split

CFG_NO_BUCKET=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_3gpu_routed_opd_rl_no_bucket_opd.py
CFG_SHUFFLED=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_2gpu_routed_rl_shuffled_labels.py
ADAPTER_NO_BUCKET=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_opd_rl_no_bucket_opd/checkpoint-step-1000/adapter
ADAPTER_SHUFFLED=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_2gpu_routed_rl_shuffled_labels/checkpoint-step-1000/adapter

mkdir -p "$OUT/no_bucket" "$OUT/shuffled"

CUDA_VISIBLE_DEVICES=0 "$PY" -m projects.pixvl_idea1.eval.eval_refseg --config "$CFG_NO_BUCKET" --adapter-path "$ADAPTER_NO_BUCKET" --schema-file "$SUBSET_ROOT/relation_2000.jsonl" --output "$OUT/no_bucket/relation.json" &
CUDA_VISIBLE_DEVICES=1 "$PY" -m projects.pixvl_idea1.eval.eval_refseg --config "$CFG_SHUFFLED" --adapter-path "$ADAPTER_SHUFFLED" --schema-file "$SUBSET_ROOT/relation_2000.jsonl" --output "$OUT/shuffled/relation.json" &
CUDA_VISIBLE_DEVICES=2 "$PY" -m projects.pixvl_idea1.eval.eval_refseg --config "$CFG_NO_BUCKET" --adapter-path "$ADAPTER_NO_BUCKET" --schema-file "$SUBSET_ROOT/geometry_2000.jsonl" --output "$OUT/no_bucket/geometry.json" &
CUDA_VISIBLE_DEVICES=3 "$PY" -m projects.pixvl_idea1.eval.eval_refseg --config "$CFG_SHUFFLED" --adapter-path "$ADAPTER_SHUFFLED" --schema-file "$SUBSET_ROOT/geometry_2000.jsonl" --output "$OUT/shuffled/geometry.json" &
CUDA_VISIBLE_DEVICES=4 "$PY" -m projects.pixvl_idea1.eval.eval_dlc --config "$CFG_NO_BUCKET" --adapter-path "$ADAPTER_NO_BUCKET" --schema-file "$SUBSET_ROOT/semantic_2000.jsonl" --output "$OUT/no_bucket/semantic.json" &
CUDA_VISIBLE_DEVICES=5 "$PY" -m projects.pixvl_idea1.eval.eval_dlc --config "$CFG_SHUFFLED" --adapter-path "$ADAPTER_SHUFFLED" --schema-file "$SUBSET_ROOT/semantic_2000.jsonl" --output "$OUT/shuffled/semantic.json" &
CUDA_VISIBLE_DEVICES=6 "$PY" -m projects.pixvl_idea1.eval.eval_refseg --config "$CFG_NO_BUCKET" --adapter-path "$ADAPTER_NO_BUCKET" --schema-file "$SUBSET_ROOT/refseg_val_2000.jsonl" --output "$OUT/no_bucket/overall.json" &
CUDA_VISIBLE_DEVICES=7 "$PY" -m projects.pixvl_idea1.eval.eval_refseg --config "$CFG_SHUFFLED" --adapter-path "$ADAPTER_SHUFFLED" --schema-file "$SUBSET_ROOT/refseg_val_2000.jsonl" --output "$OUT/shuffled/overall.json" &
wait

CUDA_VISIBLE_DEVICES=0 "$PY" -m projects.pixvl_idea1.eval.eval_dlc --config "$CFG_NO_BUCKET" --adapter-path "$ADAPTER_NO_BUCKET" --schema-file "$SUBSET_ROOT/dlc_eval_100.jsonl" --output "$OUT/no_bucket/dlc_reward.json" &
CUDA_VISIBLE_DEVICES=4 "$PY" -m projects.pixvl_idea1.eval.eval_dlc --config "$CFG_SHUFFLED" --adapter-path "$ADAPTER_SHUFFLED" --schema-file "$SUBSET_ROOT/dlc_eval_100.jsonl" --output "$OUT/shuffled/dlc_reward.json" &
wait
