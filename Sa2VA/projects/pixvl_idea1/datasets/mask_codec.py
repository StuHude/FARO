from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor

from .schema import decode_rle_mask


def _load_samtok_sam2_symbols():
    candidates = []
    sa2va_root = os.environ.get("SAMTOK_SA2VA_ROOT", "")
    if sa2va_root:
        candidates.append(Path(sa2va_root) / "projects" / "samtok" / "models" / "sam2.py")
    pixvl_root = os.environ.get("FARO_PROJECT_ROOT", os.environ.get("PIXVl_ROOT", ""))
    if pixvl_root:
        candidates.append(Path(pixvl_root) / "third_party" / "Sa2VA" / "projects" / "samtok" / "models" / "sam2.py")
    candidates.append(Path("/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/samtok/models/sam2.py"))
    sam2_path = next((path for path in candidates if path.is_file()), candidates[0])
    spec = importlib.util.spec_from_file_location("pixvl_samtok_sam2", sam2_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load SAMTok sam2 module from {sam2_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SAM2Config, module.VQ_SAM2, module.VQ_SAM2Config


SAM2Config, VQ_SAM2, VQ_SAM2Config = _load_samtok_sam2_symbols()


@dataclass
class MaskTokenCatalog:
    start_token: str
    end_token: str
    index_to_token: dict[int, str]
    codebook_size: int
    codebook_depth: int

    def build_text(self, codes: list[int]) -> str:
        return self.start_token + "".join(self.index_to_token[code] for code in codes) + self.end_token

    def extract_codes(self, text: str) -> list[int]:
        pattern = re.escape(self.start_token) + r"(.*?)" + re.escape(self.end_token)
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            return []
        inner = match.group(1)
        codes: list[int] = []
        for code, token in sorted(self.index_to_token.items(), key=lambda item: len(item[1]), reverse=True):
            inner = inner.replace(token, f" {code} ")
        for fragment in inner.split():
            if fragment.isdigit():
                codes.append(int(fragment))
        return codes[: self.codebook_depth]


class SAMTokMaskCodec:
    def __init__(
        self,
        model_name_or_path: str,
        mask_tokenizer_path: str,
        sam2_ckpt_path: str,
        codebook_size: int = 256,
        codebook_depth: int = 2,
        device: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.mask_tokenizer_path = mask_tokenizer_path
        self.sam2_ckpt_path = sam2_ckpt_path
        self.codebook_size = codebook_size
        self.codebook_depth = codebook_depth
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.catalog = self.discover_catalog(self.processor.tokenizer, codebook_size, codebook_depth)

        self.mask_tokenizer_path = self._resolve_artifact(mask_tokenizer_path)
        self.sam2_ckpt_path = self._resolve_artifact(sam2_ckpt_path)

        sam2_config = SAM2Config(ckpt_path=self.sam2_ckpt_path)
        vq_sam2_config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=codebook_size,
            codebook_depth=codebook_depth,
            shared_codebook=False,
            latent_dim=256,
        )
        self.vq_sam2 = VQ_SAM2(vq_sam2_config).to(self.device).eval()
        state = torch.load(self.mask_tokenizer_path, map_location="cpu", weights_only=False)
        self.vq_sam2.load_state_dict(state, strict=False)
        self.preprocessor = DirectResize(1024)

    @staticmethod
    def _resolve_artifact(path_or_hf: str) -> str:
        if Path(path_or_hf).exists():
            return str(Path(path_or_hf))
        if "/" in path_or_hf:
            repo_id, filename = path_or_hf.split("/", 1)
            if "/" in filename:
                repo_id = f"{repo_id}/{filename.split('/', 1)[0]}"
                filename = filename.split("/", 1)[1]
            local_path = hf_hub_download(repo_id=repo_id, filename=filename)
            return local_path
        return path_or_hf

    @staticmethod
    def discover_catalog(tokenizer: Any, codebook_size: int, codebook_depth: int) -> MaskTokenCatalog:
        vocab = tokenizer.get_vocab()
        start_token = next(token for token in vocab if token == "<|mt_start|>")
        end_token = next(token for token in vocab if token == "<|mt_end|>")
        index_to_token: dict[int, str] = {}
        for token in vocab:
            match = re.fullmatch(r"<\|mt_(\d{4})\|>", token)
            if match:
                index_to_token[int(match.group(1))] = token
        return MaskTokenCatalog(
            start_token=start_token,
            end_token=end_token,
            index_to_token=index_to_token,
            codebook_size=codebook_size,
            codebook_depth=codebook_depth,
        )

    def encode_mask(self, image: Image.Image, mask_obj: dict[str, Any]) -> list[int]:
        binary_mask = decode_rle_mask(mask_obj)
        mask_tensor = torch.from_numpy(binary_mask[None, ...]).to(self.device)

        ys, xs = np.where(binary_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return [self.codebook_size * depth for depth in range(self.codebook_depth)]

        bbox = torch.tensor(
            [[xs.min(), ys.min(), xs.max(), ys.max()]],
            dtype=torch.float32,
            device=self.device,
        )
        bbox[:, 0::2] /= float(image.width)
        bbox[:, 1::2] /= float(image.height)

        image_np = np.asarray(image.convert("RGB"))
        sam2_image = self.preprocessor.apply_image(image_np)
        sam2_tensor = torch.from_numpy(np.array(sam2_image, copy=True)).permute(2, 0, 1).contiguous().to(self.device)

        with torch.no_grad():
            output = self.vq_sam2(
                sam2_tensor.unsqueeze(0),
                [mask_tensor],
                bbox,
                reconstruct_mask=False,
            )
        quant_codes = self._flatten_codes(output.quant_codes.detach().cpu().tolist())
        remapped = [depth * self.codebook_size + int(code) for depth, code in enumerate(quant_codes)]
        return remapped

    def decode_codes(self, image: Image.Image, codes: list[int]) -> np.ndarray:
        if len(codes) != self.codebook_depth:
            return np.zeros((image.height, image.width), dtype=np.uint8)

        remapped = []
        for depth, code in enumerate(codes):
            remapped.append(max(min(int(code) - depth * self.codebook_size, self.codebook_size - 1), 0))

        image_np = np.asarray(image.convert("RGB"))
        sam2_image = self.preprocessor.apply_image(image_np)
        sam2_tensor = torch.from_numpy(np.array(sam2_image, copy=True)).permute(2, 0, 1).contiguous().to(self.device)
        quant_ids = torch.tensor([remapped], dtype=torch.long, device=self.device)
        with torch.no_grad():
            pred_masks = self.vq_sam2.forward_with_codes(sam2_tensor.unsqueeze(0), quant_ids)
        pred_masks = torch.nn.functional.interpolate(
            pred_masks,
            size=(image.height, image.width),
            mode="bilinear",
        )
        pred_masks = (pred_masks[:, 0] > 0.5).detach().cpu().numpy().astype(np.uint8)
        return pred_masks[0]

    def codes_to_text(self, codes: list[int]) -> str:
        return self.catalog.build_text(codes)

    def text_to_codes(self, text: str) -> list[int]:
        return self.catalog.extract_codes(text)

    def cache_key(self, image_path: str, mask_obj: dict[str, Any]) -> str:
        payload = json.dumps(
            {
                "model": self.model_name_or_path,
                "image_path": image_path,
                "mask": mask_obj,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _flatten_codes(codes: Any) -> list[int]:
        if isinstance(codes, torch.Tensor):
            codes = codes.detach().cpu().tolist()
        flat: list[int] = []
        if isinstance(codes, (list, tuple)):
            for item in codes:
                if isinstance(item, (list, tuple)):
                    flat.extend(SAMTokMaskCodec._flatten_codes(item))
                else:
                    flat.append(int(item))
            return flat
        return [int(codes)]


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
        return np.asarray(pil.resize((self.target_length, self.target_length)))
