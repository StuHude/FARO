#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
ACC=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/accelerate
OUT_ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3
LOG_DIR=${OUT_ROOT}/scale100k_logs

CFG_UNIFIED=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_2gpu_unified_opd_rl.py
CFG_ROUTED_RL=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_3gpu_routed_rl.py
CFG_ROUTED_OPD=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_3gpu_routed_opd_rl.py

mkdir -p "$LOG_DIR"
mkdir -p ${OUT_ROOT}/scale100k_2gpu_unified_opd_rl
mkdir -p ${OUT_ROOT}/scale100k_3gpu_routed_rl
mkdir -p ${OUT_ROOT}/scale100k_3gpu_routed_opd_rl

nohup bash -lc "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} && export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE} && export TORCH_FR_BUFFER_SIZE=${TORCH_FR_BUFFER_SIZE} && export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG} && export CUDA_VISIBLE_DEVICES=0,1 && stdbuf -oL -eL ${ACC} launch --main_process_port 30520 --num_processes 2 --mixed_precision bf16 -m projects.pixvl_idea1.trainers.joint_opd_rl_trainer --config ${CFG_UNIFIED}" > "${LOG_DIR}/unified.log" 2>&1 &
echo $! > "${LOG_DIR}/unified.pid"

nohup bash -lc "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} && export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE} && export TORCH_FR_BUFFER_SIZE=${TORCH_FR_BUFFER_SIZE} && export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG} && export CUDA_VISIBLE_DEVICES=2,3,4 && stdbuf -oL -eL ${ACC} launch --main_process_port 30522 --num_processes 3 --mixed_precision bf16 -m projects.pixvl_idea3.trainers.joint_routed_opd_rl_trainer --config ${CFG_ROUTED_RL}" > "${LOG_DIR}/routed_rl.log" 2>&1 &
echo $! > "${LOG_DIR}/routed_rl.pid"

nohup bash -lc "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} && export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE} && export TORCH_FR_BUFFER_SIZE=${TORCH_FR_BUFFER_SIZE} && export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG} && export CUDA_VISIBLE_DEVICES=5,6,7 && stdbuf -oL -eL ${ACC} launch --main_process_port 30524 --num_processes 3 --mixed_precision bf16 -m projects.pixvl_idea3.trainers.joint_routed_opd_rl_trainer --config ${CFG_ROUTED_OPD}" > "${LOG_DIR}/routed_opd.log" 2>&1 &
echo $! > "${LOG_DIR}/routed_opd.pid"
