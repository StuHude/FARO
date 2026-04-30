#!/usr/bin/env bash
set -euo pipefail

OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_eval
mkdir -p "$OUT"

nohup bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_recognition_negsup_refcoco_8gpu.sh > "$OUT/refcoco_launcher.log" 2>&1 &
nohup bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_recognition_negsup_dlc_official_8gpu.sh > "$OUT/dlc_launcher.log" 2>&1 &
nohup bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_recognition_negsup_split_8gpu.sh > "$OUT/split_launcher.log" 2>&1 &
nohup bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/finalize_recognition_negsup_eval.sh > "$OUT/finalize.log" 2>&1 &
