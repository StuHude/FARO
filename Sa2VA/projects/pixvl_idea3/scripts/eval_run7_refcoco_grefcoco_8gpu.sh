#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
MODEL=/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok
VQ=$MODEL/mask_tokenizer_256x2.pth
SAM2=$MODEL/sam2.1_hiera_large.pt
ADAPTER=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/semcovcal_routed_opd_rl_8gpu_100_run7_fast/checkpoint-step-100/adapter
CONFIG=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_semcovcal_routed_opd_rl_8gpu_500.py
REF_DATASET=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/refcoco_val.json
REF_IMAGE_ROOT=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg/refcoco/coco2014/train2014
GREF_SCHEMA=/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/grefcoco_val.jsonl

OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/run7_ckpt100_eval/refcoco_grefcoco
LOG=$OUT/launcher.log
REF_TMP=$OUT/refcoco_temp
GREF_PARTS=$OUT/gref_parts
mkdir -p "$OUT" "$REF_TMP" "$GREF_PARTS"

log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG" ; }

rm -rf "$REF_TMP"
mkdir -p "$REF_TMP"
rm -f "$GREF_PARTS"/grefcoco.part*.json

log "start refcoco official 8-way shards"
PIDS=()
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="$rank" "$PY" -m projects.samtok.evaluation.qwen3vl.qwen3vl_refcoco_padt_style_eval \
    --model_path "$MODEL" \
    --adapter_path "$ADAPTER" \
    --vq_sam2_path "$VQ" \
    --sam2_path "$SAM2" \
    --dataset "$REF_DATASET" \
    --image_folder "$REF_IMAGE_ROOT" \
    --temp_save_dir "$REF_TMP" \
    --task_id "$rank" \
    --num_tasks 8 \
    > "$OUT/refcoco_gpu${rank}.log" 2>&1 &
  PIDS+=($!)
  sleep 5
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

log "compute refcoco official metrics"
CUDA_VISIBLE_DEVICES=0 "$PY" - <<'PY' > /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/run7_ckpt100_eval/refcoco_grefcoco/refcoco_metrics.json
import json, os
from collections import defaultdict
import numpy as np
import torch
import torchvision
from pycocotools import mask as mask_utils

TEMP_DIR = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/run7_ckpt100_eval/refcoco_grefcoco/refcoco_temp"
VLM_JSON = "/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO/rec_jsons_processed/refcoco_val.json"

def rle_to_mask(rle):
    masks = []
    for r in rle:
        m = mask_utils.decode(r)
        m = np.uint8(m)
        masks.append(m)
    return np.stack(masks, axis=0)

def bbox_iou(box1, box2):
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2
    a = [x1,y1,x1+w1,y1+h1]
    b = [x2,y2,x2+w2,y2+h2]
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    union = w1*h1 + w2*h2 - inter
    return inter/union if union > 0 else 0.0

def ciou(pred, gt):
    i = np.logical_and(pred, gt).sum()
    u = np.logical_or(pred, gt).sum()
    return i/u if u > 0 else 0.0

acc = defaultdict(float)
mask_cious = defaultdict(float)

for fn in os.listdir(TEMP_DIR):
    if not fn.endswith(".json"):
        continue
    item = json.load(open(os.path.join(TEMP_DIR, fn)))
    bbox_name = item["bbox_name"]
    gt_masks = rle_to_mask(item["gt_masks"])
    pred_masks = rle_to_mask(item["prediction_masks"])
    gt_t = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in gt_masks])
    pred_t = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in pred_masks])
    try:
        gx1,gy1,gx2,gy2 = torchvision.ops.masks_to_boxes(gt_t).squeeze().cpu().numpy().tolist()
        gt_box = [gx1,gy1,gx2-gx1,gy2-gy1]
    except Exception:
        gt_box = [0,0,0,0]
    try:
        px1,py1,px2,py2 = torchvision.ops.masks_to_boxes(pred_t).squeeze().cpu().numpy().tolist()
        pred_box = [px1,py1,px2-px1,py2-py1]
    except Exception:
        pred_box = [0,0,0,0]
    iou = bbox_iou(gt_box, pred_box)
    mciou = ciou(pred_masks > 0, gt_masks > 0)
    acc[bbox_name] = max(acc[bbox_name], iou)
    mask_cious[bbox_name] = max(mask_cious[bbox_name], mciou)

all_ious = np.array(list(acc.values()))
all_cious = np.array(list(mask_cious.values()))
our_ap50 = float((all_ious >= 0.5).mean()) if len(all_ious) else 0.0
our_ciou = float(all_cious.mean()) if len(all_cious) else 0.0

vlm_ap, vlm_ciou = [], []
for item in json.load(open(VLM_JSON)):
    image_id = int(item["image"].split("_")[-1].split(".")[0])
    category = item["normal_caption"]
    key = f"{image_id}_{category}"
    vlm_ap.append(acc[key] >= 0.5)
    vlm_ciou.append(mask_cious[key])

payload = {
    "refcoco_vlmr1_rec_ap50": float(np.mean(vlm_ap)) if vlm_ap else 0.0,
    "refcoco_vlmr1_res_ciou": float(np.mean(vlm_ciou)) if vlm_ciou else 0.0,
    "refcoco_our_val_rec_ap50": our_ap50,
    "refcoco_our_val_res_ciou": our_ciou,
    "num_cases": len(acc),
}
json.dump(payload, open("/dev/stdout", "w"), indent=2)
PY

log "start grefcoco 8-way schema eval"
PIDS=()
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="$rank" "$PY" -m projects.pixvl_idea1.eval.eval_refseg \
    --config "$CONFIG" \
    --adapter-path "$ADAPTER" \
    --schema-file "$GREF_SCHEMA" \
    --output "$GREF_PARTS/grefcoco.part${rank}.json" \
    --task-id "$rank" \
    --num-tasks 8 \
    > "$OUT/grefcoco_gpu${rank}.log" 2>&1 &
  PIDS+=($!)
  sleep 3
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

log "merge grefcoco parts"
CUDA_VISIBLE_DEVICES=0 "$PY" - <<'PY' > /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/run7_ckpt100_eval/refcoco_grefcoco/grefcoco_metrics.json
import glob, json
parts = []
for path in sorted(glob.glob("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/run7_ckpt100_eval/refcoco_grefcoco/gref_parts/grefcoco.part*.json")):
    parts.append(json.load(open(path)))
results = []
for part in parts:
    results.extend(part.get("results", []))
num = len(results)
mean_ciou = sum(x["ciou"] for x in results) / max(num, 1)
ap50 = sum(1 for x in results if x["ciou"] >= 0.5) / max(num, 1)
json.dump({"num_samples": num, "mean_ciou": mean_ciou, "ap50": ap50}, open("/dev/stdout","w"), indent=2)
PY

log "refcoco+grefcoco eval done"
