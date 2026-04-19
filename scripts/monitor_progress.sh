#!/usr/bin/env bash

set -euo pipefail

LOG_FILE="${1:-/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/logs/progress_watch.log}"
mkdir -p "$(dirname "$LOG_FILE")"

while true; do
  {
    echo "=== $(date '+%F %T') ==="
    du -sh /mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co 2>/dev/null || true
    stat -c 'refcoco_plus %s' /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/sa2va_training/ref_seg_coco_plus.zip 2>/dev/null || true
    stat -c 'refcocog %s' /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/sa2va_training/ref_seg_coco_g.zip 2>/dev/null || true
    echo -n "gar_part1_arrows "
    find /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/gar/Fine-Grained-Dataset-Part1 -maxdepth 1 -type f -name 'data-*.arrow' 2>/dev/null | wc -l
    echo -n "dam_cocostuff_tars "
    find /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dam/COCOStuff/images -maxdepth 1 -type f -name '*.tar' 2>/dev/null | wc -l
    echo -n "dam_lvis_tars "
    find /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dam/LVIS/images -maxdepth 1 -type f -name '*.tar' 2>/dev/null | wc -l
    echo -n "dam_paco_tars "
    find /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dam/PACO/images -maxdepth 1 -type f -name '*.tar' 2>/dev/null | wc -l
    ps -ef | rg 'download_refseg_only.sh|download_dam_selected.sh|download_gar_part1_only.sh|aria2c.*ref_seg_coco_plus|aria2c.*ref_seg_coco_g|aria2c.*LVIS|aria2c.*PACO|prepare_gar_data.py|conda create -y -p /mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa' || true
    echo
  } >> "$LOG_FILE"
  sleep 300
done
