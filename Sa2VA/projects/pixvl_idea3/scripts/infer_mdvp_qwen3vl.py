from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from peft import PeftModel
from pycocotools import mask as mask_utils
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from projects.samtok.models import VQ_SAM2, VQ_SAM2Config, SAM2Config, DirectResize


MT_START_TOKEN = "<|mt_start|>"
MT_END_TOKEN = "<|mt_end|>"
MT_CONTEXT_TOKEN = "<|mt_{}|>"
CODEBOOK_SIZE = 256
CODEBOOK_DEPTH = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference of Qwen/SAMTok-style models on MDVP-Bench.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--sam2-path", required=True)
    parser.add_argument("--vq-sam2-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--anno-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def decode_mask(mask_obj: dict, ori_height: int, ori_width: int) -> np.ndarray:
    if isinstance(mask_obj, dict):
        if isinstance(mask_obj["counts"], list):
            mask_obj = mask_utils.frPyObjects(mask_obj, ori_height, ori_width)
        m = mask_utils.decode(mask_obj)
        return m.astype(np.uint8).squeeze()
    rles = mask_utils.frPyObjects(mask_obj, ori_height, ori_width)
    rle = mask_utils.merge(rles)
    m = mask_utils.decode(rle).astype(np.uint8).squeeze()
    return m


def encode_codes_text(quant_codes: list[int]) -> str:
    return MT_START_TOKEN + "".join(MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes) + MT_END_TOKEN


def build_models(args: argparse.Namespace):
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn_impl = "sdpa"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model = model.cuda().eval()
    processor = AutoProcessor.from_pretrained(args.processor_path, trust_remote_code=True)

    sam2_config = SAM2Config(ckpt_path=args.sam2_path)
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )
    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()
    state = torch.load(args.vq_sam2_path, map_location="cpu")
    vq_sam2.load_state_dict(state, strict=False)
    sampler = DirectResize(1024)
    return model, processor, vq_sam2, sampler


def encode_mask_tokens(
    image: Image.Image,
    binary_mask: np.ndarray,
    vq_sam2: VQ_SAM2,
    sampler: DirectResize,
) -> tuple[str, float, tuple[int, int, int, int]]:
    ori_width, ori_height = image.size
    sam2_image = np.array(image)
    sam2_image = sampler.apply_image(sam2_image)
    sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
    sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

    masks = torch.stack([torch.from_numpy(np.ascontiguousarray(binary_mask.copy()))])
    boxes = torchvision.ops.masks_to_boxes(masks)
    x1, y1, x2, y2 = boxes.squeeze().cpu().numpy().tolist()
    boxes_w = x2 - x1
    boxes_h = y2 - y1
    boxes_area = boxes_h * boxes_w
    image_area = ori_height * ori_width
    boxes_occupied_ratio = boxes_area / image_area

    whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
    boxes = (boxes / whwh).to(vq_sam2.device)
    masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]

    with torch.no_grad():
        vq_out = vq_sam2(
            sam2_pixel_values,
            masks,
            boxes,
            reconstruct_mask=False,
        )
    quant_codes = vq_out.quant_codes.squeeze().cpu().numpy().astype(np.int32).tolist()
    quant_codes = [depth_idx * CODEBOOK_SIZE + quant_code for depth_idx, quant_code in enumerate(quant_codes)]
    return encode_codes_text(quant_codes), boxes_occupied_ratio, (int(x1), int(y1), int(x2), int(y2))


def build_messages(image_path: str, image: Image.Image, region_text: str, crop_payload: tuple[Image.Image, str] | None):
    with open(image_path, "rb") as f:
        global_b64 = base64.b64encode(f.read()).decode()

    if crop_payload is not None:
        crop_image, crop_region_text = crop_payload
        buf = io.BytesIO()
        crop_image.save(buf, format="JPEG")
        buf.seek(0)
        crop_b64 = base64.b64encode(buf.read()).decode()
        question = f"Given a detailed description of this region {region_text}. Zoom in with the perspective as "
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{global_b64}"},
                    {"type": "text", "text": question},
                    {"type": "image", "image": f"data:image/jpeg;base64,{crop_b64}"},
                    {"type": "text", "text": f", {crop_region_text}."},
                ],
            }
        ]

    question = f"Given a detailed description of this region {region_text}."
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{global_b64}"},
                {"type": "text", "text": question},
            ],
        }
    ]


def main() -> None:
    args = parse_args()
    model, processor, vq_sam2, sampler = build_models(args)
    with open(args.anno_file, "r") as f:
        eval_samples = json.load(f)

    model_outputs = []
    indices = list(range(len(eval_samples)))[args.task_id :: args.num_tasks]

    for idx in indices:
        eval_sample = eval_samples[idx]
        image_file = eval_sample["image_path"]
        image_path = os.path.join(args.image_root, image_file)
        image = Image.open(image_path).convert("RGB")
        ori_width, ori_height = image.size

        binary_mask = decode_mask(eval_sample["mask_rle"], ori_height, ori_width)
        global_mask_tokens_str, occupied_ratio, (x1, y1, x2, y2) = encode_mask_tokens(image, binary_mask, vq_sam2, sampler)

        crop_payload = None
        if occupied_ratio < 0.2:
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            if bbox_w < 140:
                x1 = x1 - (140 - bbox_w) // 2
                x2 = x2 + (140 - bbox_w) // 2
            if bbox_h < 140:
                y1 = y1 - (140 - bbox_h) // 2
                y2 = y2 + (140 - bbox_h) // 2
            x1 = int(max(0, x1))
            x2 = int(min(ori_width, x2))
            y1 = int(max(0, y1))
            y2 = int(min(ori_height, y2))

            cropped_image = image.crop((x1, y1, x2, y2))
            crop_width, crop_height = cropped_image.size
            if crop_width > crop_height and crop_width < 280:
                ratio = 280 / crop_height
                new_height = 280
                new_width = int(crop_width * ratio)
                cropped_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            elif crop_height > crop_width and crop_height < 280:
                ratio = 280 / crop_width
                new_width = 280
                new_height = int(crop_height * ratio)
                cropped_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            elif crop_height == crop_width and crop_width < 280:
                cropped_image = cropped_image.resize((280, 280), Image.Resampling.LANCZOS)

            cropped_mask = binary_mask[y1:y2, x1:x2]
            if cropped_image.size != (x2 - x1, y2 - y1):
                resized = torch.nn.functional.interpolate(
                    torch.from_numpy(cropped_mask[None, None].astype(np.float32)),
                    size=(cropped_image.size[1], cropped_image.size[0]),
                    mode="bilinear",
                )
                cropped_mask = (resized[0, 0].numpy() > 0.5).astype(np.uint8)
            crop_mask_tokens_str, _, _ = encode_mask_tokens(cropped_image, cropped_mask, vq_sam2, sampler)
            crop_payload = (cropped_image, crop_mask_tokens_str)

        messages = build_messages(image_path, image, global_mask_tokens_str, crop_payload)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            top_p=1.0,
        )
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        pred_caption = output_text[0].replace("<|im_end|>", "").strip()
        model_outputs.append(
            {
                "image_path": image_file,
                "caption": pred_caption,
                "gt": eval_sample["caption"],
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(model_outputs, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
