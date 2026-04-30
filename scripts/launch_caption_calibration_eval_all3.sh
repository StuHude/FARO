#!/usr/bin/env bash
set -euo pipefail

OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_eval
mkdir -p "$OUT"

nohup bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_caption_calibration_refcoco_8gpu.sh > "$OUT/refcoco_launcher.log" 2>&1 &
echo $! > "$OUT/refcoco_launcher.pid"

nohup bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_caption_calibration_dlc_official_8gpu.sh > "$OUT/dlc_launcher.log" 2>&1 &
echo $! > "$OUT/dlc_launcher.pid"

nohup bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_caption_calibration_split_8gpu.sh > "$OUT/split_launcher.log" 2>&1 &
echo $! > "$OUT/split_launcher.pid"

nohup bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/finalize_caption_calibration_eval.sh > "$OUT/finalize.log" 2>&1 &
echo $! > "$OUT/finalize.pid"
