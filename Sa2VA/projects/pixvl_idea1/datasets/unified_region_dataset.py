from __future__ import annotations

import json
import math
import os
import random
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Sampler

from .mask_codec import SAMTokMaskCodec
from .overlay_utils import build_overlay_image
from .schema import decode_rle_mask, load_jsonl

ImageFile.LOAD_TRUNCATED_IMAGES = True


def identity_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


class MaskCodeCache:
    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS mask_codes (cache_key TEXT PRIMARY KEY, codes_json TEXT NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str) -> list[int] | None:
        row = self.conn.execute("SELECT codes_json FROM mask_codes WHERE cache_key = ?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: str, codes: list[int]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO mask_codes(cache_key, codes_json) VALUES(?, ?)",
            (key, json.dumps(codes)),
        )
        self.conn.commit()


class UnifiedRegionDataset(Dataset):
    def __init__(
        self,
        schema_files: Iterable[str],
        model_name_or_path: str,
        mask_tokenizer_path: str,
        sam2_ckpt_path: str,
        cache_path: str,
        task_mix: dict[str, float],
        source_mix: dict[str, float] | None,
        prompt_templates: dict[str, str],
        overlay_cfg: dict[str, Any],
        bucket_mix: dict[str, float] | None = None,
        visual_token_filter: dict[str, Any] | None = None,
        codec_device: str | None = None,
    ) -> None:
        self.records: list[dict[str, Any]] = []
        for schema_file in schema_files:
            path = Path(schema_file)
            if path.exists():
                self.records.extend(load_jsonl(path))

        self.codec = SAMTokMaskCodec(
            model_name_or_path=model_name_or_path,
            mask_tokenizer_path=mask_tokenizer_path,
            sam2_ckpt_path=sam2_ckpt_path,
            device=codec_device,
        )
        self.cache = MaskCodeCache(cache_path)
        self.task_mix = task_mix
        self.bucket_mix: dict[str, float] = bucket_mix or {}
        self.source_mix: dict[str, float] = source_mix or {}
        self.prompt_templates = prompt_templates
        self.overlay_cfg = overlay_cfg
        self.visual_token_filter = visual_token_filter or {}
        self.record_by_id = {record["id"]: record for record in self.records}
        self._debug_enabled = os.environ.get("PIXVL_DATASET_DEBUG", "0") == "1"
        self._debug_budget = 8
        self.visual_token_unit = self._discover_visual_token_unit()
        self._apply_visual_token_filter()

    def _debug(self, message: str) -> None:
        if not self._debug_enabled or self._debug_budget <= 0:
            return
        self._debug_budget -= 1
        print(f"[dataset-debug pid={os.getpid()}] {message}", flush=True)

    def __len__(self) -> int:
        return len(self.records)

    def _discover_visual_token_unit(self) -> int:
        image_processor = getattr(self.codec.processor, "image_processor", None)
        patch_size = int(getattr(image_processor, "patch_size", 16) or 16)
        merge_size = int(getattr(image_processor, "merge_size", 2) or 2)
        return max(patch_size * merge_size, 1)

    def _estimate_visual_tokens(self, record: dict[str, Any]) -> int:
        height, width = HomogeneousTaskBatchSampler._mask_hw(record)
        unit = self.visual_token_unit
        return max(math.ceil(height / unit) * math.ceil(width / unit), 1)

    def _apply_visual_token_filter(self) -> None:
        if not self.visual_token_filter.get("enabled", False):
            return
        if not self.records:
            return

        multiplier = float(self.visual_token_filter.get("max_ratio_to_avg", 1.5))
        estimates = [self._estimate_visual_tokens(record) for record in self.records]
        avg_tokens = sum(estimates) / max(len(estimates), 1)
        threshold = avg_tokens * multiplier

        filtered_records: list[dict[str, Any]] = []
        dropped = 0
        for record, estimate in zip(self.records, estimates):
            record = dict(record)
            meta = dict(record.get("meta") or {})
            meta["estimated_visual_tokens"] = estimate
            record["meta"] = meta
            if estimate > threshold:
                dropped += 1
                continue
            filtered_records.append(record)

        self.records = filtered_records
        self.record_by_id = {record["id"]: record for record in self.records}
        print(
            f"[pixvl] visual_token_filter enabled unit={self.visual_token_unit} "
            f"avg={avg_tokens:.2f} threshold={threshold:.2f} "
            f"kept={len(self.records)} dropped={dropped}",
            flush=True,
        )

    def _ensure_codes(self, record: dict[str, Any], image: Image.Image) -> list[int]:
        cache_key = self.codec.cache_key(record["image_path"], record["mask"])
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._debug(f"ensure_codes hit id={record['id']}")
            return cached
        self._debug(f"ensure_codes miss id={record['id']}")
        codes = self.codec.encode_mask(image, record["mask"])
        self.cache.set(cache_key, codes)
        self._debug(f"ensure_codes encoded id={record['id']}")
        return codes

    def _ensure_codes_for_mask(
        self,
        image_path: str,
        mask_obj: dict[str, Any],
        image: Image.Image,
    ) -> list[int]:
        cache_key = self.codec.cache_key(image_path, mask_obj)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        codes = self.codec.encode_mask(image, mask_obj)
        self.cache.set(cache_key, codes)
        return codes

    def __getitem__(self, index: int) -> dict[str, Any]:
        for offset in range(len(self.records)):
            record = self.records[(index + offset) % len(self.records)]
            try:
                self._debug(f"getitem start index={index} record={record['id']} task={record['task']} source={record['source']}")
                image = Image.open(record["image_path"]).convert("RGB")
                self._debug(f"getitem image_opened record={record['id']}")
                if record["task"] in {"existence", "direct_sft"}:
                    prompt = record.get("query") or record.get("prompt") or "the described target"
                    answer = record.get("answer") or record.get("caption") or "No target."
                    task = record["task"]
                    return {
                        "id": record["id"], "task": task, "source": record.get("source", task),
                        "route_bucket": task, "image_path": record["image_path"], "image": image,
                        "overlay_image": image, "mask": None, "mask_binary": None,
                        "query": prompt, "caption": answer, "split": record.get("split"), "gt_codes": [],
                        "gt_mask_tokens": "", "aux_gt_codes": None, "aux_gt_mask_tokens": None,
                        "prompt_text": (
                            f"Does the image contain this target? {prompt}\nAnswer yes or no."
                            if task == "existence" else str(prompt)
                        ),
                        "answer_text": answer, "meta": record.get("meta", {}),
                    }
                # Standard SAMTok RefCOCO exports contain the supervised mask
                # token sequence but may not retain the original COCO RLE.
                # Decode that label with the original SAMTok codec so ordinary
                # on-policy RL can use the same geometry reward as RLE rows.
                token_only = record.get("mask_tokens") or record.get("gt_mask_tokens")
                no_target = bool((record.get("meta") or {}).get("no_target", False))
                if token_only and not record.get("mask"):
                    codes = self.codec.text_to_codes(str(token_only))
                    mask = self.codec.decode_codes(image, codes)
                    mask_obj = None
                else:
                    mask_obj = record["mask"]
                    mask = decode_rle_mask(mask_obj)
                overlay = build_overlay_image(
                    image=image,
                    mask=mask,
                    darken_alpha=float(self.overlay_cfg.get("darken_alpha", 0.4)),
                    boundary_px=int(self.overlay_cfg.get("boundary_px", 2)),
                    boundary_color=self.overlay_cfg.get("boundary_color", [255, 64, 64]),
                )
                self._debug(f"getitem overlay_done record={record['id']}")
                if no_target and record["task"] == "refseg":
                    codes = []
                elif token_only and not record.get("mask"):
                    codes = self.codec.text_to_codes(str(token_only))
                else:
                    codes = self._ensure_codes(record, image)
                mask_tokens = self.codec.codes_to_text(codes) if codes else ""
                self._debug(f"getitem codes_ready record={record['id']}")
                aux_codes = None
                aux_mask_tokens = None
                if record["task"] == "refseg":
                    prompt_text = self.prompt_templates["refseg"].format(query=record["query"])
                    answer_text = "No target." if no_target else mask_tokens
                else:
                    aux_mask = (record.get("meta") or {}).get("aux_mask")
                    if aux_mask is not None:
                        aux_codes = self._ensure_codes_for_mask(record["image_path"], aux_mask, image)
                        aux_mask_tokens = self.codec.codes_to_text(aux_codes)
                        prompt_template = self.prompt_templates.get(
                            "relation_maskcap",
                            "Region A: {mask_tokens_a}\nRegion B: {mask_tokens_b}\nDescribe the relation of Region A to Region B in just a few words.",
                        )
                        prompt_text = prompt_template.format(
                            mask_tokens_a=mask_tokens,
                            mask_tokens_b=aux_mask_tokens,
                        )
                    else:
                        aux_codes = None
                        aux_mask_tokens = None
                        meta = record.get("meta") or {}
                        prompt_key = meta.get("prompt_key", "maskcap")
                        template = self.prompt_templates.get(prompt_key, self.prompt_templates["maskcap"])
                        prompt_text = template.format(mask_tokens=mask_tokens)
                    answer_text = record["caption"]

                return {
                    "id": record["id"],
                    "pair_id": record.get("pair_id", record["id"]),
                    "task": record["task"],
                    "source": record["source"],
                    "route_bucket": (record.get("meta") or {}).get("failure_route", "default"),
                    "image_path": record["image_path"],
                    "image": image,
                    "overlay_image": overlay,
                    "mask": mask_obj if mask_obj is not None else {"format": "decoded_tokens", "size": [int(mask.shape[0]), int(mask.shape[1])]},
                    "mask_binary": mask,
                    "query": record.get("query"),
                    "caption": record.get("caption"),
                    "split": record.get("split"),
                    "gt_codes": codes,
                    "gt_mask_tokens": mask_tokens,
                    "aux_gt_codes": aux_codes,
                    "aux_gt_mask_tokens": aux_mask_tokens,
                    "prompt_text": prompt_text,
                    "answer_text": answer_text,
                    "meta": record.get("meta", {}),
                }
            except Exception as exc:
                self._debug(
                    f"getitem exception record={record.get('id', 'unknown')} "
                    f"type={exc.__class__.__name__} msg={exc}"
                )
                continue
        raise RuntimeError("Failed to fetch any valid sample from UnifiedRegionDataset")


class HomogeneousTaskBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: UnifiedRegionDataset, batch_size: int, seed: int = 0) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.num_replicas = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
        self.group_size = self.batch_size * self.num_replicas
        self.task_to_indices: dict[str, list[int]] = {"refseg": [], "maskcap": []}
        self.grouped_indices: dict[tuple[str, str, str, int, int, int, int], list[int]] = {}
        for idx, record in enumerate(dataset.records):
            self.task_to_indices.setdefault(record["task"], []).append(idx)
            key = (
                record["task"],
                record["source"],
                self._route_bucket(record),
                self._length_bucket(record),
                self._image_bucket(record),
                self._pixel_bucket(record),
                self._aspect_bucket(record),
            )
            self.grouped_indices.setdefault(key, []).append(idx)
        self.task_to_group_keys: dict[str, list[tuple[str, str, str, int, int, int, int]]] = {}
        self.task_source_to_group_keys: dict[tuple[str, str], list[tuple[str, str, str, int, int, int, int]]] = {}
        for key in self.grouped_indices:
            task, source, *_ = key
            self.task_to_group_keys.setdefault(task, []).append(key)
            self.task_source_to_group_keys.setdefault((task, source), []).append(key)
        self.total_sampler_batches = self._compute_total_sampler_batches()
        self.next_epoch = 0
        self.batch_offset = 0

    @staticmethod
    def _length_bucket(record: dict[str, Any]) -> int:
        text = record.get("query") or record.get("caption") or ""
        return min(len(text) // 24, 7)

    @staticmethod
    def _route_bucket(record: dict[str, Any]) -> str:
        meta = record.get("meta") or {}
        return str(meta.get("failure_route", "default"))

    @staticmethod
    def _mask_hw(record: dict[str, Any]) -> tuple[int, int]:
        mask = record.get("mask") or {}
        size = mask.get("size") if isinstance(mask, dict) else None
        if isinstance(size, list) and len(size) == 2:
            return int(size[0]), int(size[1])
        meta = record.get("meta") or {}
        height = meta.get("height")
        width = meta.get("width")
        if height is not None and width is not None:
            return int(height), int(width)
        return 1024, 1024

    @classmethod
    def _image_bucket(cls, record: dict[str, Any]) -> int:
        height, width = cls._mask_hw(record)
        longest = max(height, width)
        boundaries = [512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1792, 2048]
        for bucket, boundary in enumerate(boundaries):
            if longest <= boundary:
                return bucket
        return len(boundaries)

    @classmethod
    def _pixel_bucket(cls, record: dict[str, Any]) -> int:
        height, width = cls._mask_hw(record)
        mp = (height * width) / 1_000_000.0
        boundaries = [0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5]
        for bucket, boundary in enumerate(boundaries):
            if mp <= boundary:
                return bucket
        return len(boundaries)

    @classmethod
    def _aspect_bucket(cls, record: dict[str, Any]) -> int:
        height, width = cls._mask_hw(record)
        ratio = max(width / max(height, 1), height / max(width, 1))
        if ratio < 1.2:
            return 0
        if ratio < 1.8:
            return 1
        return 2

    def _build_group_pools(self, rng: random.Random) -> dict[tuple[str, str, str, int, int, int, int], list[int]]:
        pools = {key: indices.copy() for key, indices in self.grouped_indices.items()}
        for indices in pools.values():
            rng.shuffle(indices)
        return pools

    @staticmethod
    def _allocate_counts(weights: dict[str, float], total: int) -> dict[str, int]:
        if total <= 0 or not weights:
            return {}
        normalized = {key: max(float(value), 0.0) for key, value in weights.items()}
        positive = {key: value for key, value in normalized.items() if value > 0.0}
        if not positive:
            keys = list(normalized.keys())
            counts = {key: 0 for key in keys}
            for idx in range(total):
                counts[keys[idx % len(keys)]] += 1
            return counts

        weight_sum = sum(positive.values())
        raw = {key: total * value / weight_sum for key, value in positive.items()}
        counts = {key: int(math.floor(value)) for key, value in raw.items()}
        remaining = total - sum(counts.values())
        ranked = sorted(
            positive.keys(),
            key=lambda key: (raw[key] - counts[key], positive[key], key),
            reverse=True,
        )
        for idx in range(remaining):
            counts[ranked[idx % len(ranked)]] += 1
        for key in normalized:
            counts.setdefault(key, 0)
        return counts

    def _choose_group_key(
        self,
        pools: dict[tuple[str, str, str, int, int, int, int], list[int]],
        rng: random.Random,
        task: str | None = None,
        source: str | None = None,
    ) -> tuple[str, str, str, int, int, int, int] | None:
        available_keys = [
            key
            for key, indices in pools.items()
            if len(indices) >= 1
            and (task is None or key[0] == task)
            and (source is None or key[1] == source)
        ]
        if not available_keys:
            return None
        weights: list[float] = []
        for key in available_keys:
            task, source, route_bucket, _, image_bucket, pixel_bucket, _ = key
            task_weight = max(float(self.dataset.task_mix.get(task, 1.0)), 0.0)
            source_weight = max(float(self.dataset.source_mix.get(source, 1.0)), 0.0) if task == "maskcap" else 1.0
            bucket_weight = max(float(self.dataset.bucket_mix.get(route_bucket, 1.0)), 0.0)
            image_weight = 1.0 + float(image_bucket) * 0.2 + float(pixel_bucket) * 0.25
            remaining_weight = math.sqrt(len(pools[key]))
            weights.append(task_weight * source_weight * bucket_weight * image_weight * remaining_weight)
        if sum(weights) <= 0.0:
            return rng.choice(available_keys)
        return rng.choices(available_keys, weights=weights, k=1)[0]

    def _compute_total_sampler_batches(self) -> int:
        total_records = sum(len(indices) for indices in self.grouped_indices.values())
        return (total_records // self.group_size) * self.num_replicas

    def _task_available_counts(
        self,
        pools: dict[tuple[str, str, str, int, int, int, int], list[int]],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for key, indices in pools.items():
            if not indices:
                continue
            counts[key[0]] = counts.get(key[0], 0) + len(indices)
        return counts

    def _source_available_counts(
        self,
        pools: dict[tuple[str, str, str, int, int, int, int], list[int]],
        task: str,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for key, indices in pools.items():
            if not indices or key[0] != task:
                continue
            counts[key[1]] = counts.get(key[1], 0) + len(indices)
        return counts

    def _build_global_block(
        self,
        pools: dict[tuple[str, str, str, int, int, int, int], list[int]],
        rng: random.Random,
    ) -> list[int] | None:
        total_available = sum(len(indices) for indices in pools.values())
        if total_available < self.group_size:
            return None

        task_available = self._task_available_counts(pools)
        if not task_available:
            return None

        eligible_tasks = {
            task: count
            for task, count in task_available.items()
            if count >= self.group_size
        }
        if not eligible_tasks:
            return None

        task_weights = {
            task: float(self.dataset.task_mix.get(task, 1.0))
            for task in eligible_tasks
        }
        block: list[int] = []
        if sum(task_weights.values()) <= 0.0:
            chosen_task = rng.choice(list(eligible_tasks.keys()))
        else:
            chosen_task = rng.choices(list(task_weights.keys()), weights=list(task_weights.values()), k=1)[0]

        if chosen_task == "maskcap":
            source_available = self._source_available_counts(pools, chosen_task)
            source_weights = {
                source: float(self.dataset.source_mix.get(source, 1.0))
                for source in source_available
            }
            source_counts = self._allocate_counts(source_weights, self.group_size)
            for source, source_count in source_counts.items():
                for _ in range(source_count):
                    chosen_key = self._choose_group_key(pools, rng, task=chosen_task, source=source)
                    if chosen_key is None:
                        break
                    block.append(pools[chosen_key].pop())

        while len(block) < self.group_size:
            chosen_key = self._choose_group_key(pools, rng, task=chosen_task)
            if chosen_key is None:
                break
            block.append(pools[chosen_key].pop())

        if len(block) < self.group_size:
            return None

        rng.shuffle(block)
        return block

    def _build_epoch_batches(self, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + epoch)
        pools = self._build_group_pools(rng)
        batches: list[list[int]] = []
        while True:
            block = self._build_global_block(pools, rng)
            if block is None:
                break
            for start in range(0, self.group_size, self.batch_size):
                batches.append(block[start : start + self.batch_size])
        return batches

    def set_start_step(self, step: int) -> None:
        if self.total_sampler_batches <= 0:
            self.next_epoch = 0
            self.batch_offset = 0
            return
        consumed_sampler_batches = step * self.num_replicas
        self.next_epoch = consumed_sampler_batches // self.total_sampler_batches
        self.batch_offset = consumed_sampler_batches % self.total_sampler_batches

    def state_dict(self) -> dict[str, int]:
        return {
            "next_epoch": int(self.next_epoch),
            "batch_offset": int(self.batch_offset),
            "num_replicas": int(self.num_replicas),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.next_epoch = int(state.get("next_epoch", 0))
        self.batch_offset = int(state.get("batch_offset", 0))

    def __iter__(self):
        epoch = self.next_epoch
        start_offset = self.batch_offset
        epoch_batches = self._build_epoch_batches(epoch)
        for batch_idx in range(start_offset, len(epoch_batches)):
            self.next_epoch = epoch
            self.batch_offset = batch_idx + 1
            yield epoch_batches[batch_idx]
        self.next_epoch = epoch + 1
        self.batch_offset = 0

    def __len__(self) -> int:
        return self.total_sampler_batches


class FixedRecordOrderBatchSampler(Sampler[list[int]]):
    """Yield schema records in order for exact-multiset curriculum controls."""

    def __init__(self, dataset: UnifiedRegionDataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
        self.total_sampler_batches = (
            len(dataset.records) // (self.batch_size * self.num_replicas)
        ) * self.num_replicas
        self.next_epoch = 0
        self.batch_offset = 0

    def set_start_step(self, step: int) -> None:
        consumed = int(step) * self.num_replicas
        if self.total_sampler_batches <= 0:
            self.next_epoch = 0
            self.batch_offset = 0
            return
        self.next_epoch = consumed // self.total_sampler_batches
        self.batch_offset = consumed % self.total_sampler_batches

    def state_dict(self) -> dict[str, int]:
        return {
            "next_epoch": self.next_epoch,
            "batch_offset": self.batch_offset,
            "num_replicas": self.num_replicas,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.next_epoch = int(state.get("next_epoch", 0))
        self.batch_offset = int(state.get("batch_offset", 0))

    def __iter__(self):
        start = self.batch_offset * self.batch_size
        stop = self.total_sampler_batches * self.batch_size
        for offset in range(start, stop, self.batch_size):
            self.batch_offset = offset // self.batch_size + 1
            yield list(range(offset, offset + self.batch_size))
        self.next_epoch += 1
        self.batch_offset = 0

    def __len__(self) -> int:
        return self.total_sampler_batches
