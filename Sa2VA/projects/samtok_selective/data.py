from __future__ import annotations

import json
import os
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Protocol

from PIL import Image, ImageFile
from torch.utils.data import Dataset, Sampler


ImageFile.LOAD_TRUNCATED_IMAGES = True


class MaskEncoder(Protocol):
    def encode_mask(self, image: Image.Image, mask_obj: dict[str, Any]) -> list[int]: ...

    def codes_to_text(self, codes: list[int]) -> str: ...

    def cache_key(self, image_path: str, mask_obj: dict[str, Any]) -> str: ...


def load_refseg_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != "refseg":
                raise ValueError(f"Only refseg rows are allowed ({path}:{line_number})")
            for key in ("id", "pair_id", "image_path", "query", "mask", "meta"):
                if key not in row:
                    raise ValueError(f"Missing {key} ({path}:{line_number})")
            mask = row["mask"]
            if mask.get("format") != "rle" or not isinstance(mask.get("size"), list):
                raise ValueError(f"Expected COCO RLE mask ({path}:{line_number})")
            rows.append(row)
    if not rows:
        raise ValueError(f"No refseg rows in {path}")
    return rows


def validate_pair_parity(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int | None = None,
    expected_no_target_rows: int | None = None,
) -> dict[str, int]:
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Refseg row IDs must be unique")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["pair_id"])].append(row)
    for pair_id, pair in groups.items():
        outcomes = [bool(row["meta"].get("no_target", False)) for row in pair]
        if len(pair) != 2 or sorted(outcomes) != [False, True]:
            raise ValueError(f"Pair {pair_id} must contain one positive and one no-target row")
    no_target_rows = sum(bool(row["meta"].get("no_target", False)) for row in rows)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(rows)}")
    if expected_no_target_rows is not None and no_target_rows != expected_no_target_rows:
        raise ValueError(f"Expected {expected_no_target_rows} no-target rows, found {no_target_rows}")
    return {"rows": len(rows), "pairs": len(groups), "no_target_rows": no_target_rows}


class MaskCodeCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=120.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS mask_codes (cache_key TEXT PRIMARY KEY, codes_json TEXT NOT NULL)"
        )
        self.connection.commit()

    def get(self, key: str) -> list[int] | None:
        item = self.connection.execute(
            "SELECT codes_json FROM mask_codes WHERE cache_key = ?", (key,)
        ).fetchone()
        return None if item is None else [int(code) for code in json.loads(item[0])]

    def set(self, key: str, codes: list[int]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO mask_codes(cache_key, codes_json) VALUES (?, ?)",
            (key, json.dumps(codes)),
        )
        self.connection.commit()


class SelectiveRefSegDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        jsonl_path: str | Path,
        prompt_template: str,
        codec: MaskEncoder,
        cache_path: str | Path,
        *,
        expected_rows: int | None = None,
        expected_no_target_rows: int | None = None,
    ) -> None:
        self.rows = load_refseg_jsonl(jsonl_path)
        self.parity = validate_pair_parity(
            self.rows,
            expected_rows=expected_rows,
            expected_no_target_rows=expected_no_target_rows,
        )
        self.prompt_template = prompt_template
        self.codec = codec
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        cache_path = Path(cache_path)
        self.cache = MaskCodeCache(cache_path.with_name(f"{cache_path.stem}.rank{rank}{cache_path.suffix}"))
        self.pair_to_indices: dict[str, tuple[int, int]] = {}
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            grouped[str(row["pair_id"])].append(index)
        for pair_id, indices in grouped.items():
            positive = next(index for index in indices if not self._is_no_target(self.rows[index]))
            negative = next(index for index in indices if self._is_no_target(self.rows[index]))
            self.pair_to_indices[pair_id] = (positive, negative)

    @staticmethod
    def _is_no_target(row: dict[str, Any]) -> bool:
        return bool(row["meta"].get("no_target", False))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        no_target = self._is_no_target(row)
        codes: list[int] = []
        if not no_target:
            key = self.codec.cache_key(row["image_path"], row["mask"])
            cached = self.cache.get(key)
            if cached is None:
                cached = self.codec.encode_mask(image, row["mask"])
                self.cache.set(key, cached)
            codes = cached
        answer = "No target." if no_target else self.codec.codes_to_text(codes)
        return {
            "id": str(row["id"]),
            "pair_id": str(row["pair_id"]),
            "image": image,
            "image_path": str(row["image_path"]),
            "prompt_text": self.prompt_template.format(query=row["query"]),
            "answer_text": answer,
            "no_target": no_target,
            "mask": row["mask"],
            "codes": codes,
        }


class PairedBatchSampler(Sampler[list[int]]):
    """Deterministic batches containing complete positive/no-target pairs."""

    def __init__(self, dataset: SelectiveRefSegDataset, pairs_per_batch: int, seed: int) -> None:
        if pairs_per_batch < 1:
            raise ValueError("pairs_per_batch must be positive")
        self.dataset = dataset
        self.pairs_per_batch = pairs_per_batch
        self.seed = seed

    def __iter__(self) -> Iterator[list[int]]:
        pair_ids = sorted(self.dataset.pair_to_indices)
        random.Random(self.seed).shuffle(pair_ids)
        for start in range(0, len(pair_ids), self.pairs_per_batch):
            selected = pair_ids[start : start + self.pairs_per_batch]
            if len(selected) < self.pairs_per_batch:
                break
            batch: list[int] = []
            for pair_id in selected:
                batch.extend(self.dataset.pair_to_indices[pair_id])
            yield batch

    def __len__(self) -> int:
        return len(self.dataset.pair_to_indices) // self.pairs_per_batch


class RegisteredPairedBatchSampler(Sampler[list[int]]):
    """Replay a preregistered pair-ID schedule without reshuffling it."""

    def __init__(
        self, dataset: SelectiveRefSegDataset, pair_id_batches: list[list[str]]
    ) -> None:
        if not pair_id_batches or any(len(batch) < 1 for batch in pair_id_batches):
            raise ValueError("registered paired schedule must contain nonempty batches")
        missing = sorted(
            {
                str(pair_id)
                for batch in pair_id_batches
                for pair_id in batch
                if str(pair_id) not in dataset.pair_to_indices
            }
        )
        if missing:
            raise ValueError(f"registered schedule contains unknown pair IDs: {missing[:3]}")
        flat = [str(pair_id) for batch in pair_id_batches for pair_id in batch]
        if len(flat) != len(set(flat)):
            raise ValueError("registered schedule pair IDs must be unique")
        self.dataset = dataset
        self.pair_id_batches = [list(map(str, batch)) for batch in pair_id_batches]

    def __iter__(self) -> Iterator[list[int]]:
        for pair_ids in self.pair_id_batches:
            batch: list[int] = []
            for pair_id in pair_ids:
                batch.extend(self.dataset.pair_to_indices[pair_id])
            yield batch

    def __len__(self) -> int:
        return len(self.pair_id_batches)


def identity_collate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return samples
