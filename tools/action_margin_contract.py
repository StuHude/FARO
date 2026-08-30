#!/usr/bin/env python3
"""Strict, dependency-free contracts for standalone SAMTok action margins."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EXPECTED_ROWS = 512
EXPECTED_ROWS_PER_LABEL = EXPECTED_ROWS // 2
MARGIN_SCHEMA_VERSION = 3
MARGIN_FORMAT = "samtok_action_margin"


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_items(items: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(item.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class SchemaContract:
    path: Path
    sha256: str
    row_ids_sha256: str
    rows: tuple[dict[str, Any], ...]

    @property
    def source_image_ids(self) -> frozenset[str]:
        return frozenset(str(row["meta"]["source_image_id"]) for row in self.rows)

    @property
    def source_image_paths(self) -> frozenset[str]:
        return frozenset(str(row.get("source_image") or row.get("image_path") or "") for row in self.rows)


def load_schema_contract(path: str | Path) -> SchemaContract:
    path = Path(path).resolve()
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"schema must contain exactly {EXPECTED_ROWS} rows, got {len(rows)}: {path}")

    ids: list[str] = []
    pair_labels: dict[str, set[bool]] = {}
    labels: list[bool] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("task") != "refseg":
            raise ValueError(f"schema row {index} is not a refseg object")
        row_id = row.get("id")
        pair_id = row.get("pair_id")
        meta = row.get("meta")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"schema row {index} has no string id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"schema row {index} has no string pair_id")
        if not isinstance(meta, dict):
            raise ValueError(f"schema row {index} has no metadata object")
        no_target = meta.get("no_target")
        if type(no_target) is not bool:
            raise ValueError(f"schema row {index} has no boolean no_target label")
        source_image_id = meta.get("source_image_id")
        if source_image_id is None or not str(source_image_id):
            raise ValueError(f"schema row {index} has no source_image_id")
        expected_suffix = "negative" if no_target else "positive"
        if not row_id.endswith(f"-{expected_suffix}"):
            raise ValueError(f"schema row {index} id/label mismatch: {row_id}")
        ids.append(row_id)
        labels.append(no_target)
        pair_labels.setdefault(pair_id, set()).add(no_target)

    if len(set(ids)) != EXPECTED_ROWS:
        raise ValueError("schema IDs must be unique")
    if labels.count(True) != EXPECTED_ROWS_PER_LABEL or labels.count(False) != EXPECTED_ROWS_PER_LABEL:
        raise ValueError("schema must contain exactly 256 positive and 256 no-target rows")
    if len(pair_labels) != EXPECTED_ROWS_PER_LABEL or any(value != {False, True} for value in pair_labels.values()):
        raise ValueError("schema must contain 256 complete positive/no-target pairs")
    return SchemaContract(
        path=path,
        sha256=sha256_file(path),
        row_ids_sha256=sha256_text_items(ids),
        rows=rows,
    )


def assert_disjoint_source_images(train: SchemaContract, holdout: SchemaContract) -> None:
    id_overlap = train.source_image_ids & holdout.source_image_ids
    path_overlap = train.source_image_paths & holdout.source_image_paths
    if id_overlap or path_overlap:
        preview = sorted(id_overlap or path_overlap)[:8]
        raise ValueError(
            "train/holdout source_image overlap "
            f"(IDs={len(id_overlap)}, paths={len(path_overlap)}): {preview}"
        )
    if train.sha256 == holdout.sha256:
        raise ValueError("train and holdout schemas must have different SHA256 digests")


def _validate_adapter_manifest(payload: object, path: Path) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"invalid standalone provenance schema: {path}")
    base = payload.get("base_checkpoint")
    if not isinstance(base, str) or "samtok" not in base.lower():
        raise ValueError(f"provenance does not identify a SAMTok base checkpoint: {path}")
    for section, key in (("config", "sha256"), ("source", "sha256"), ("data", "sha256")):
        item = payload.get(section)
        if not isinstance(item, dict) or not _is_sha256(item.get(key)):
            raise ValueError(f"provenance has no valid {section}.{key}: {path}")
    data = payload["data"]
    if not _is_sha256(data.get("row_ids_sha256")) or not isinstance(data.get("row_ids"), list):
        raise ValueError(f"provenance has incomplete data identity: {path}")


def validate_formal_adapter(adapter: str | Path, output_root: str | Path) -> dict[str, str]:
    adapter = Path(adapter).resolve()
    output_root = Path(output_root).resolve()
    if adapter.name != "adapter" or adapter.parent.parent != output_root:
        raise ValueError(f"adapter must be a direct formal run below {output_root}: {adapter}")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        artifact = adapter / filename
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f"adapter is missing nonempty {filename}: {adapter}")

    run_dir = adapter.parent
    metrics_path = run_dir / "metrics.json"
    provenance_path = run_dir / "provenance_manifest.json"
    if not metrics_path.is_file() or not provenance_path.is_file():
        raise ValueError(f"formal adapter requires metrics.json and provenance_manifest.json: {adapter}")
    metrics = _read_json(metrics_path)
    if not isinstance(metrics, dict) or metrics.get("status") != "finished":
        raise ValueError(f"adapter metrics are not finished: {metrics_path}")
    steps = metrics.get("steps_completed")
    if type(steps) is not int or steps < 1:
        raise ValueError(f"adapter metrics have no positive steps_completed: {metrics_path}")
    _validate_adapter_manifest(_read_json(provenance_path), provenance_path)
    return {
        "adapter": str(adapter),
        "metrics_sha256": sha256_file(metrics_path),
        "provenance_sha256": sha256_file(provenance_path),
    }


def validate_margin_shards(
    root: str | Path,
    split: str,
    policy: str,
    count: int,
) -> tuple[list[dict[str, Any]], SchemaContract, dict[str, Any]]:
    if count < 1:
        raise ValueError("num_shards must be positive")
    root = Path(root)
    payloads: list[dict[str, Any]] = []
    for rank in range(count):
        path = root / f"{split}_{policy}_part_{rank}.json"
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"margin shard is not an object: {path}")
        payloads.append(payload)

    first = payloads[0]
    schema = load_schema_contract(str(first.get("schema", "")))
    shared_keys = (
        "format",
        "schema_version",
        "schema",
        "schema_sha256",
        "schema_row_ids_sha256",
        "schema_row_count",
        "model",
        "adapter",
        "adapter_metrics_sha256",
        "adapter_provenance_sha256",
        "policy",
        "num_tasks",
        "scoring",
    )
    shared = {key: first.get(key) for key in shared_keys}
    if shared["format"] != MARGIN_FORMAT or shared["schema_version"] != MARGIN_SCHEMA_VERSION:
        raise ValueError(f"unsupported margin schema for {split}/{policy}")
    if shared["schema_sha256"] != schema.sha256:
        raise ValueError(f"schema SHA mismatch for {split}/{policy}")
    if shared["schema_row_ids_sha256"] != schema.row_ids_sha256:
        raise ValueError(f"schema row-ID SHA mismatch for {split}/{policy}")
    if shared["schema_row_count"] != EXPECTED_ROWS:
        raise ValueError(f"invalid schema row count for {split}/{policy}")
    if shared["policy"] != policy or shared["num_tasks"] != count:
        raise ValueError(f"policy/shard metadata mismatch for {split}/{policy}")
    if not _is_sha256(shared["adapter_metrics_sha256"]) or not _is_sha256(
        shared["adapter_provenance_sha256"]
    ):
        raise ValueError(f"invalid adapter provenance SHA for {split}/{policy}")
    scoring = shared["scoring"]
    if not isinstance(scoring, dict) or scoring.get("canonical_null_text") != "No target.":
        raise ValueError(f"invalid scoring metadata for {split}/{policy}")
    if not scoring.get("null_token_ids") or not scoring.get("mask_start_token_ids"):
        raise ValueError(f"scoring metadata has empty candidate IDs for {split}/{policy}")

    records: list[dict[str, Any]] = []
    for rank, payload in enumerate(payloads):
        if {key: payload.get(key) for key in shared_keys} != shared:
            raise ValueError(f"inconsistent shard metadata for {split}/{policy}, rank {rank}")
        if payload.get("task_id") != rank:
            raise ValueError(f"task_id mismatch for {split}/{policy}, rank {rank}")
        shard_records = payload.get("records")
        if not isinstance(shard_records, list):
            raise ValueError(f"records are missing for {split}/{policy}, rank {rank}")
        expected_indices = list(range(rank, EXPECTED_ROWS, count))
        actual_indices = [row.get("index") for row in shard_records if isinstance(row, dict)]
        if actual_indices != expected_indices or len(shard_records) != len(expected_indices):
            raise ValueError(f"index coverage mismatch for {split}/{policy}, rank {rank}")
        records.extend(shard_records)

    records.sort(key=lambda row: int(row["index"]))
    for index, (record, schema_row) in enumerate(zip(records, schema.rows)):
        meta = schema_row["meta"]
        if type(record.get("index")) is not int or type(record.get("no_target")) is not bool:
            raise ValueError(f"record index/label types are invalid for {split}/{policy}, index {index}")
        if any(type(record.get(key)) is not str for key in ("id", "pair_id", "source_image_id")):
            raise ValueError(f"record identity types are invalid for {split}/{policy}, index {index}")
        expected = {
            "index": index,
            "id": str(schema_row["id"]),
            "pair_id": str(schema_row["pair_id"]),
            "no_target": bool(meta["no_target"]),
            "source_image_id": str(meta["source_image_id"]),
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"record/schema identity mismatch for {split}/{policy}, index {index}")
        try:
            null_log_prob = float(record["null_action_log_prob"])
            mask_log_prob = float(record["mask_start_log_prob"])
            margin = float(record["margin"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid margin fields for {split}/{policy}, index {index}") from error
        if not all(math.isfinite(value) for value in (null_log_prob, mask_log_prob, margin)):
            raise ValueError(f"non-finite margin for {split}/{policy}, index {index}")
        if not math.isclose(margin, null_log_prob - mask_log_prob, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"margin arithmetic mismatch for {split}/{policy}, index {index}")
    return records, schema, shared


def validate_policy_alignment(
    left: list[dict[str, Any]], right: list[dict[str, Any]], split: str
) -> None:
    identity_keys = ("index", "id", "pair_id", "no_target", "source_image_id")
    left_identity = [tuple(row[key] for key in identity_keys) for row in left]
    right_identity = [tuple(row[key] for key in identity_keys) for row in right]
    if left_identity != right_identity:
        raise ValueError(f"cross-policy record alignment mismatch for {split}")


def load_aligned_eval(path: str | Path, schema: SchemaContract) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"eval payload is not an object: {path}")
    section = payload.get("refseg_overall", payload)
    records = section.get("records") if isinstance(section, dict) else None
    if not isinstance(records, list) or len(records) != EXPECTED_ROWS:
        raise ValueError(f"eval must contain exactly {EXPECTED_ROWS} records: {path}")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"eval records must be objects: {path}")
    expected_ids = [str(row["id"]) for row in schema.rows]
    actual_ids = [str(row.get("id", "")) for row in records]
    if len(set(actual_ids)) != EXPECTED_ROWS or set(actual_ids) != set(expected_ids):
        raise ValueError(f"eval records do not exactly align to schema IDs: {path}")
    by_id = {str(record["id"]): record for record in records}
    for index, schema_row in enumerate(schema.rows):
        record = by_id[str(schema_row["id"])]
        expected_exists = not bool(schema_row["meta"]["no_target"])
        if type(record.get("truth_exists")) is not bool or record["truth_exists"] != expected_exists:
            raise ValueError(f"eval label mismatch at index {index}: {path}")
        if type(record.get("explicit_null")) is not bool:
            raise ValueError(f"eval explicit_null is not boolean at index {index}: {path}")
    return {row_id: by_id[row_id] for row_id in expected_ids}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    adapters = subparsers.add_parser("validate-adapters")
    adapters.add_argument("--output-root", required=True)
    adapters.add_argument("adapter", nargs="+")
    schemas = subparsers.add_parser("validate-schemas")
    schemas.add_argument("--train", required=True)
    schemas.add_argument("--holdout", required=True)
    args = parser.parse_args()
    if args.command == "validate-adapters":
        result = [validate_formal_adapter(path, args.output_root) for path in args.adapter]
    else:
        train = load_schema_contract(args.train)
        holdout = load_schema_contract(args.holdout)
        assert_disjoint_source_images(train, holdout)
        result = {"train_sha256": train.sha256, "holdout_sha256": holdout.sha256}
    print(json.dumps({"status": "ok", "result": result}, sort_keys=True))


if __name__ == "__main__":
    main()
