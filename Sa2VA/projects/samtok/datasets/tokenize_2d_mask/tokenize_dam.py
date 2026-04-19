import os
import sys
import random
import copy
from PIL import Image
import numpy as np
import torch
import torchvision
from pycocotools import mask as mask_utils
import json
import tqdm
import uuid

from projects.samtok.models import VQ_SAM2, VQ_SAM2Config, SAM2Config, DirectResize

def decode_mask(object_masks, ori_height, ori_width):
    binary_masks = []
    for object_mask in object_masks:
        if isinstance(object_mask, dict):
            if isinstance(object_mask["counts"], list):
                # convert to compressed RLE
                object_mask = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            m = mask_utils.decode(object_mask)
            m = m.astype(np.uint8).squeeze()
        elif object_mask:
            rles = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            rle = mask_utils.merge(rles)
            m = mask_utils.decode(rle).astype(np.uint8).squeeze()
        else:
            m = np.zeros((ori_height, ori_width), dtype=np.uint8)
        binary_masks.append(m)
    return binary_masks

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union


QUESTION_LIST = [
    "Given a detailed description of this region {SEG}. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "Provide a thorough description of this region {SEG}. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "Describe the region {SEG} in detail. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "Provide detailed information about this region {SEG}. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "Provide a detailed caption of this region {SEG}. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "{SEG}\nGive a detailed description of the masked region. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "{SEG}\nProvide a detailed description of the masked region. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "{SEG}\nDescribe the masked area comprehensively. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "{SEG}\nWhat are the details of the masked area? Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
]

GLOBAL_QUESTION_LIST = [
    "Given a detailed description of this region {SEG}.",
    "Provide a thorough description of this region {SEG}.",
    "Describe the region {SEG} in detail.",
    "Provide detailed information about this region {SEG}.",
    "Provide a detailed caption of this region {SEG}.",
    "{SEG}\nGive a detailed description of the masked region.",
    "{SEG}\nProvide a detailed description of the masked region.",
    "{SEG}\nDescribe the masked area comprehensively.",
    "{SEG}\nWhat are the details of the masked area?",
]

def main(task_id):

    # build mask tokenizer
    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data_256x2/dam"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "dam"
    
    sam2_config = SAM2Config(
        cfg_path="sam2.1_hiera_l.yaml",
        ckpt_path="zhouyik/Qwen3-VL-8B-SAMTok/sam2.1_hiera_large.pt",
    )
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )
    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()
    state = torch.load("zhouyik/Qwen3-VL-8B-SAMTok/mask_tokenizer_256x2.pth", map_location="cpu")
    vq_sam2.load_state_dict(state)

    sam2_image_processor = DirectResize(1024)

    # tokenize 2d mask
    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    dam_data_root = "./data/dam_data"
    split_name_list = ["COCOStuff", "LVIS", "Mapillary", "OpenImages", "PACO", "SAM"]
    split_image_folder = {
        'COCOStuff': './data/',
        'LVIS': './data/',
        'Mapillary': './data/dam_data/Mapillary/images/',
        'OpenImages': './data/dam_data/OpenImages/images/',
        'PACO': './data/coco/',
        'SAM': './data/dam_data/SAM/images/'
    }

    for split_name in split_name_list:
        
        split_path = os.path.join(dam_data_root, split_name)
        annotation_path = os.path.join(split_path, "annotations.json")

        with open(annotation_path, 'r') as f:
            annotations_dict = json.load(f)

        rows = len(annotations_dict)
        chunk_size = (rows+7) // 8
        _start_ = task_id * chunk_size
        _end_ = _start_ + chunk_size
        _end_ = rows if _end_ > rows else _end_
        
        index = 0

        all_items = list(annotations_dict.items())
        subset = all_items[_start_:_end_]
        for _, data_dict in tqdm.tqdm(subset):
            for item in data_dict:
                caption = item['caption']
                if split_name == 'OpenImages':
                    image_id = item['img_id']
                    image_file = f"{image_id}.jpg"
                else:
                    image_file = os.path.basename(item['image'])
                    if '.jpg' in image_file:
                        image_id = image_file.split('.')[0]
                    elif '.png' in image_file:
                        image_id = image_file.split('.')[0]
                    else:
                        raise ValueError(f"Unsupported image format: {image_file}")
                ann_id = item['ann_id']
                
                if os.path.exists(os.path.join(temp_save_root, f"{image_id}_{ann_id}_{split_name}.json")):
                    print("file exists.............")
                    continue
                if split_name in ['COCOStuff', 'LVIS', 'PACO']:
                    image_path = os.path.join(split_image_folder[split_name], item['image'])
                elif split_name in ['Mapillary', 'OpenImages', 'SAM']:
                    image_path = os.path.join(split_image_folder[split_name], image_file)
                image = Image.open(image_path).convert('RGB')
                ori_width, ori_height = image.size

                sam2_image = np.array(image)
                sam2_image = sam2_image_processor.apply_image(sam2_image)
                sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
                sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

                binary_masks = decode_mask([item['mask_rle']], ori_height, ori_width)

                masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
                boxes = torchvision.ops.masks_to_boxes(masks)
                x1, y1, x2, y2 = boxes.squeeze().cpu().numpy().tolist()
                boxes_w = boxes[:, 2] - boxes[:, 0]
                boxes_h = boxes[:, 3] - boxes[:, 1]
                boxes_area = boxes_h * boxes_w
                image_area = ori_height * ori_width
                boxes_occupied_ratio = boxes_area / image_area

                whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
                boxes = boxes / whwh
                boxes = boxes.to(vq_sam2.device)
                masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
                
                with torch.no_grad():
                    vq_sam2_output = vq_sam2(
                        sam2_pixel_values,
                        masks,
                        boxes,
                        reconstruct_mask=False,
                    )

                quant_codes = vq_sam2_output.quant_codes.squeeze().detach().cpu().numpy().astype(np.int32).tolist()
                remap_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(quant_codes)]
                quant_codes = remap_quant_codes
                if boxes_occupied_ratio[0].item() > 0.2:
                    mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN
                    question = random.choice(GLOBAL_QUESTION_LIST).format(SEG=mask_tokens_str)
                    question = "<image>\n" + question
                    conversation = []
                    conversation.append({'from': 'human', 'value': question})
                    conversation.append({'from': 'gpt', 'value': caption})
                    ret_data_dict = {
                        'image': [image_path],
                        'conversations': conversation,
                    }
                    shard_items.append(ret_data_dict)
                    count += 1

                    if count % shard_size == 0:
                        shard_idx += 1
                        out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-chunk{task_id}-{shard_idx:05d}.json")
                        with open(out_path, "w") as f:
                            json.dump(shard_items, f)
                        shard_items.clear()
                        print(f"[SAVE] {out_path} ({count} items)", flush=True)
                    continue

                # zoom in mask and image
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

                # resize the short edge
                if crop_width > crop_height and crop_width < 280:
                    ratio = 280 / crop_height
                    new_height = 280
                    new_width = int(crop_width * ratio)
                    resized_crop_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                elif crop_height > crop_width and crop_height < 280:
                    ratio = 280 / crop_width
                    new_width = 280
                    new_height = int(crop_height * ratio)
                    resized_crop_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    new_height = new_width = None
                    resized_crop_image = None

                if resized_crop_image is None:
                    cropped_sam2_image = np.array(cropped_image)
                    cropped_sam2_image = sam2_image_processor.apply_image(cropped_sam2_image)
                    cropped_sam2_pixel_values = torch.from_numpy(cropped_sam2_image).permute(2, 0, 1).contiguous()
                    cropped_sam2_pixel_values = cropped_sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
                else:
                    cropped_sam2_image = np.array(resized_crop_image)
                    cropped_sam2_image = sam2_image_processor.apply_image(cropped_sam2_image)
                    cropped_sam2_pixel_values = torch.from_numpy(cropped_sam2_image).permute(2, 0, 1).contiguous()
                    cropped_sam2_pixel_values = cropped_sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

                cropped_masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy()[y1:y2, x1:x2])) for x in binary_masks])
                assert cropped_masks.shape[-2] == crop_height and cropped_masks.shape[-1] == crop_width

                if resized_crop_image is not None:
                    resized_crop_masks = torch.nn.functional.interpolate(cropped_masks.unsqueeze(0), size=(new_height, new_width), mode='bilinear')
                    resized_crop_masks = resized_crop_masks[0] > 0.5
                    cropped_masks = resized_crop_masks
                crop_height, crop_width = cropped_masks.shape[-2:]
                cropped_boxes = torchvision.ops.masks_to_boxes(cropped_masks)
                crop_whwh = torch.as_tensor([[crop_width, crop_height, crop_width, crop_height]])
                cropped_boxes = cropped_boxes / crop_whwh
                cropped_boxes = cropped_boxes.to(vq_sam2.device)
                cropped_masks = [m.unsqueeze(0).to(vq_sam2.device) for m in cropped_masks]

                with torch.no_grad():
                    cropped_vq_sam2_output = vq_sam2(
                        cropped_sam2_pixel_values,
                        cropped_masks,
                        cropped_boxes,
                        reconstruct_mask=True,
                    )
                
                crop_quant_codes = cropped_vq_sam2_output.quant_codes.squeeze().detach().cpu().numpy().astype(np.int32).tolist()
                remap_crop_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(crop_quant_codes)]
                crop_quant_codes = remap_crop_quant_codes

                mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN
                crop_mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in crop_quant_codes]) + MT_END_TOKEN
                question = random.choice(QUESTION_LIST).format(SEG=mask_tokens_str, ZOOM_IN_SEG=crop_mask_tokens_str)
                question = "<image>\n" + question

                conversation = []
                conversation.append({'from': 'human', 'value': question})
                conversation.append({'from': 'gpt', 'value': caption})

                # save crop image
                random_uuid = uuid.uuid4()
                crop_image_file = f"./data/dam_data/dam_crop_images/{image_id}_{random_uuid}.jpg"
                if resized_crop_image is not None:
                    resized_crop_image.save(crop_image_file)
                else:
                    cropped_image.save(crop_image_file)
                    
                ret_data_dict = {
                    'image': [image_path, crop_image_file],
                    'conversations': conversation,
                }

                shard_items.append(ret_data_dict)
                count += 1

                if count % shard_size == 0:
                    shard_idx += 1
                    out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-chunk{task_id}-{shard_idx:05d}.json")
                    with open(out_path, "w") as f:
                        json.dump(shard_items, f)
                    shard_items.clear()
                    print(f"[SAVE] {out_path} ({count} items)", flush=True)

            if shard_items:
                shard_idx += 1
                out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-chunk{task_id}-{shard_idx:05d}.json")
                with open(out_path, "w") as f:
                    json.dump(shard_items, f)
                shard_items.clear()
                print(f"[SAVE] {out_path} ({count} items)", flush=True)

if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)
