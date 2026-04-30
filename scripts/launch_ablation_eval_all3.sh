#!/usr/bin/env bash
set -euo pipefail

mkdir -p /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8

nohup bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_ablation_refcoco_8gpu.sh > /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8/refcoco_launcher.log 2>&1 &
echo $! > /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8/refcoco_launcher.pid

nohup bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_ablation_dlc_official_8gpu.sh > /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8/dlc_launcher.log 2>&1 &
echo $! > /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8/dlc_launcher.pid

nohup bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_ablation_split_eval_8gpu.sh > /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8/split_launcher.log 2>&1 &
echo $! > /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8/split_launcher.pid
