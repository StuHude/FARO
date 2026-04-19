#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from projects.pixvl_idea1.datasets.mask_codec import SAMTokMaskCodec
from projects.pixvl_idea1.datasets.schema import decode_rle_mask, load_jsonl
from projects.pixvl_idea1.rewards.seg_reward import compute_ciou
from projects.pixvl_idea1.trainers.common import (
    build_model_bundle,
    build_prompt_and_answer_ids,
    clean_generated_text,
    generate_answer,
    load_config,
    move_inputs_to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--refseg-schema", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/refseg_train.jsonl")
    parser.add_argument("--maskcap-schema", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/dam_train.jsonl")
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    refseg_sample = load_jsonl(args.refseg_schema)[0]
    maskcap_sample = load_jsonl(args.maskcap_schema)[0]

    codec = SAMTokMaskCodec(
        model_name_or_path=cfg["model"]["processor_name_or_path"],
        mask_tokenizer_path=cfg["model"]["mask_tokenizer_path"],
        sam2_ckpt_path=cfg["model"]["sam2_ckpt_path"],
    )

    image = Image.open(refseg_sample["image_path"]).convert("RGB")
    gt_mask = decode_rle_mask(refseg_sample["mask"])
    codes = codec.encode_mask(image, refseg_sample["mask"])
    pred_mask = codec.decode_codes(image, codes)
    codec_ciou = compute_ciou(pred_mask, gt_mask)
    print({"codec_codes": codes, "codec_ciou": codec_ciou, "codec_shape": list(pred_mask.shape)}, flush=True)

    if args.skip_model:
        return

    model, processor = build_model_bundle(cfg, trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    refseg_prompt = cfg["data"]["prompts"]["refseg"].format(query=refseg_sample["query"])
    refseg_answer = codec.codes_to_text(codes)
    refseg_prompt_inputs, _ = build_prompt_and_answer_ids(processor, image, refseg_prompt, refseg_answer)
    refseg_prompt_inputs = move_inputs_to_device(refseg_prompt_inputs, device)
    _, refseg_text = generate_answer(model, processor, refseg_prompt_inputs, cfg["generation"]["refseg"])
    print({"refseg_generation": refseg_text}, flush=True)

    image_cap = Image.open(maskcap_sample["image_path"]).convert("RGB")
    cap_codes = codec.encode_mask(image_cap, maskcap_sample["mask"])
    cap_prompt = cfg["data"]["prompts"]["maskcap"].format(mask_tokens=codec.codes_to_text(cap_codes))
    cap_prompt_inputs, _ = build_prompt_and_answer_ids(processor, image_cap, cap_prompt, maskcap_sample["caption"])
    cap_prompt_inputs = move_inputs_to_device(cap_prompt_inputs, device)
    _, cap_text = generate_answer(model, processor, cap_prompt_inputs, cfg["generation"]["maskcap"])
    print({"maskcap_generation": clean_generated_text(cap_text), "maskcap_gt": maskcap_sample["caption"]}, flush=True)


if __name__ == "__main__":
    main()
