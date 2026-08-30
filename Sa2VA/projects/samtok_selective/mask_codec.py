from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


RUNTIME_PATH = Path(__file__).resolve().parents[1] / "samtok" / "demo" / "gradio" / "sam2.py"
PINNED_RUNTIME_SHA256 = "480cb6df91d5ae65622af5ee020762f0ede174e54c719a8af7f54e0cf56b864e"


def decode_rle_mask(mask_obj: dict[str, Any]) -> np.ndarray:
    from pycocotools import mask as mask_utils

    decoded = mask_utils.decode({"counts": mask_obj["counts"], "size": mask_obj["size"]})
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded.astype(np.uint8)


class DirectResize:
    def __init__(self, target_length: int = 1024) -> None:
        self.target_length = int(target_length)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(
            Image.fromarray(image.astype(np.uint8), mode="RGB").resize(
                (self.target_length, self.target_length)
            )
        )


def _runtime_symbols():
    if hashlib.sha256(RUNTIME_PATH.read_bytes()).hexdigest() != PINNED_RUNTIME_SHA256:
        raise RuntimeError(f"Pinned SAMTok codec runtime changed: {RUNTIME_PATH}")
    from projects.samtok.demo.gradio.sam2 import SAM2Config, VQ_SAM2, VQ_SAM2Config

    return SAM2Config, VQ_SAM2, VQ_SAM2Config


class SAMTokMaskCodec:
    """Mask VQ codec bound to the committed original SAMTok runtime."""

    def __init__(
        self,
        model_name_or_path: str,
        mask_tokenizer_path: str,
        sam2_ckpt_path: str,
        codebook_size: int = 256,
        codebook_depth: int = 2,
        device: str | None = None,
    ) -> None:
        from transformers import AutoProcessor

        self.model_name_or_path = str(Path(model_name_or_path).resolve())
        self.mask_tokenizer_path = str(Path(mask_tokenizer_path).resolve())
        self.sam2_ckpt_path = str(Path(sam2_ckpt_path).resolve())
        for path in (self.model_name_or_path, self.mask_tokenizer_path, self.sam2_ckpt_path):
            if not Path(path).exists():
                raise FileNotFoundError(path)
        self.codebook_size = int(codebook_size)
        self.codebook_depth = int(codebook_depth)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.catalog = self.discover_catalog(self.processor.tokenizer)
        SAM2Config, VQ_SAM2, VQ_SAM2Config = _runtime_symbols()
        sam2_config = SAM2Config(ckpt_path=self.sam2_ckpt_path)
        config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=self.codebook_size,
            codebook_depth=self.codebook_depth,
            shared_codebook=False,
            latent_dim=256,
        )
        self.vq_sam2 = VQ_SAM2(config).to(self.device).eval()
        state = torch.load(self.mask_tokenizer_path, map_location="cpu", weights_only=False)
        self.vq_sam2.load_state_dict(state, strict=False)
        self.preprocessor = DirectResize(1024)

    @staticmethod
    def discover_catalog(tokenizer: Any) -> dict[str, Any]:
        vocab = tokenizer.get_vocab()
        start_token = "<|mt_start|>"
        end_token = "<|mt_end|>"
        if start_token not in vocab or end_token not in vocab:
            raise ValueError("SAMTok mask boundary tokens are missing")
        index_to_token: dict[int, str] = {}
        for token in vocab:
            match = re.fullmatch(r"<\|mt_(\d{4})\|>", token)
            if match:
                index_to_token[int(match.group(1))] = token
        return {"start": start_token, "end": end_token, "index_to_token": index_to_token}

    def codes_to_text(self, codes: list[int]) -> str:
        catalog = self.catalog
        tokens = catalog["index_to_token"]
        if len(codes) != self.codebook_depth:
            raise ValueError(f"Expected {self.codebook_depth} mask codes, got {len(codes)}")
        return catalog["start"] + "".join(tokens[int(code)] for code in codes) + catalog["end"]

    def text_to_codes(self, text: str) -> list[int]:
        catalog = self.catalog
        match = re.search(re.escape(catalog["start"]) + r"(.*?)" + re.escape(catalog["end"]), text)
        if not match:
            return []
        reverse = {value: key for key, value in catalog["index_to_token"].items()}
        inner = match.group(1)
        for token, code in sorted(reverse.items(), key=lambda item: len(item[0]), reverse=True):
            inner = inner.replace(token, f" {code} ")
        return [int(fragment) for fragment in inner.split() if fragment.isdigit()][: self.codebook_depth]

    @staticmethod
    def _flatten_codes(value: Any) -> list[int]:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            result: list[int] = []
            for item in value:
                result.extend(SAMTokMaskCodec._flatten_codes(item))
            return result
        return [int(value)]

    def encode_mask(self, image: Image.Image, mask_obj: dict[str, Any]) -> list[int]:
        binary = decode_rle_mask(mask_obj)
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return [self.codebook_size * depth for depth in range(self.codebook_depth)]
        mask_tensor = torch.from_numpy(binary[None, ...]).to(self.device)
        bbox = torch.tensor([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=torch.float32, device=self.device)
        bbox[:, 0::2] /= float(image.width)
        bbox[:, 1::2] /= float(image.height)
        image_array = np.asarray(image.convert("RGB"))
        resized = self.preprocessor.apply_image(image_array)
        pixels = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).contiguous().to(self.device)
        with torch.no_grad():
            output = self.vq_sam2(pixels.unsqueeze(0), [mask_tensor], bbox, reconstruct_mask=False)
        quant_codes = self._flatten_codes(output.quant_codes)
        return [depth * self.codebook_size + int(code) for depth, code in enumerate(quant_codes[: self.codebook_depth])]

    def decode_codes(self, image: Image.Image, codes: list[int]) -> np.ndarray:
        if len(codes) != self.codebook_depth:
            return np.zeros((image.height, image.width), dtype=np.uint8)
        _, _, _ = _runtime_symbols()
        remapped = [max(min(int(code) - depth * self.codebook_size, self.codebook_size - 1), 0) for depth, code in enumerate(codes)]
        resized = self.preprocessor.apply_image(np.asarray(image.convert("RGB")))
        pixels = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).contiguous().to(self.device)
        quant_ids = torch.tensor([remapped], dtype=torch.long, device=self.device)
        with torch.no_grad():
            pred_masks = self.vq_sam2.forward_with_codes(pixels.unsqueeze(0), quant_ids)
        pred_masks = F.interpolate(pred_masks, size=(image.height, image.width), mode="bilinear")
        return (pred_masks[:, 0] > 0.5).detach().cpu().numpy().astype(np.uint8)[0]

    def cache_key(self, image_path: str, mask_obj: dict[str, Any]) -> str:
        payload = json.dumps({"image": image_path, "mask": mask_obj}, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()
