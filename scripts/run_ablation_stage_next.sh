#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"

ACC=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/accelerate

CFG_NO_BUCKET_OPD=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_3gpu_routed_opd_rl_no_bucket_opd.py
CFG_SHUFFLED=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_3gpu_routed_rl_shuffled_labels.py

LOG_DIR=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_logs
mkdir -p "$LOG_DIR"

nohup bash -lc "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} && export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE} && export TORCH_FR_BUFFER_SIZE=${TORCH_FR_BUFFER_SIZE} && export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG} && export CUDA_VISIBLE_DEVICES=0,1,2 && stdbuf -oL -eL ${ACC} launch --main_process_port 30620 --num_processes 3 --mixed_precision bf16 -m projects.pixvl_idea3.trainers.joint_routed_opd_rl_trainer --config ${CFG_NO_BUCKET_OPD}" > "${LOG_DIR}/no_bucket_opd.log" 2>&1 &
echo $! > "${LOG_DIR}/no_bucket_opd.pid"

nohup bash -lc "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} && export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE} && export TORCH_FR_BUFFER_SIZE=${TORCH_FR_BUFFER_SIZE} && export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG} && export CUDA_VISIBLE_DEVICES=3,4,5 && stdbuf -oL -eL ${ACC} launch --main_process_port 30622 --num_processes 3 --mixed_precision bf16 -m projects.pixvl_idea3.trainers.joint_routed_opd_rl_trainer --config ${CFG_SHUFFLED}" > "${LOG_DIR}/shuffled_labels.log" 2>&1 &
echo $! > "${LOG_DIR}/shuffled_labels.pid"
