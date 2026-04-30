#!/usr/bin/env bash
set -euo pipefail
mkdir -p /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/grefcoco
mkdir -p /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/ref_adv_s
HF_ENDPOINT=https://hf-mirror.com hf download --repo-type dataset FudanCVL/gRefCOCO --local-dir /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/grefcoco --max-workers 8
HF_ENDPOINT=https://hf-mirror.com hf download --repo-type dataset dddraxxx/ref-adv-s --local-dir /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/ref_adv_s --max-workers 8
