from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from tools.action_margin_contract import (
    MARGIN_FORMAT,
    MARGIN_SCHEMA_VERSION,
    assert_disjoint_source_images,
    load_aligned_eval,
    load_schema_contract,
    validate_formal_adapter,
    validate_margin_shards,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_schema(path: Path, image_prefix: str) -> list[dict]:
    rows = []
    for pair_index in range(256):
        for no_target in (False, True):
            label = "negative" if no_target else "positive"
            rows.append(
                {
                    "id": f"row-{pair_index:03d}-{label}",
                    "pair_id": f"pair-{pair_index:03d}",
                    "task": "refseg",
                    "image_path": f"/{image_prefix}/{pair_index}-{label}.jpg",
                    "query": "object",
                    "meta": {
                        "no_target": no_target,
                        "source_image_id": f"{image_prefix}-{pair_index}-{label}",
                    },
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def write_margin_shards(
    root: Path,
    split: str,
    policy: str,
    schema_path: Path,
    margins: list[float],
    count: int = 2,
) -> None:
    schema = load_schema_contract(schema_path)
    scoring = {
        "canonical_null_text": "No target.",
        "null_token_ids": [1, 2, 3],
        "mask_start_tokens": ["<|mt_start|>"],
        "mask_start_token_ids": [4],
        "margin_definition": "log_p_first_null_token_minus_logsumexp_mask_start",
    }
    for rank in range(count):
        records = []
        for index in range(rank, 512, count):
            row = schema.rows[index]
            margin = margins[index]
            records.append(
                {
                    "index": index,
                    "id": row["id"],
                    "pair_id": row["pair_id"],
                    "no_target": row["meta"]["no_target"],
                    "source_image_id": row["meta"]["source_image_id"],
                    "null_action_log_prob": margin,
                    "mask_start_log_prob": 0.0,
                    "margin": margin,
                }
            )
        payload = {
            "format": MARGIN_FORMAT,
            "schema_version": MARGIN_SCHEMA_VERSION,
            "schema": str(schema.path),
            "schema_sha256": schema.sha256,
            "schema_row_ids_sha256": schema.row_ids_sha256,
            "schema_row_count": 512,
            "model": "/checkpoint/SAMTok",
            "adapter": f"/standalone/{policy}/adapter",
            "adapter_metrics_sha256": "b" * 64,
            "adapter_provenance_sha256": "c" * 64,
            "policy": policy,
            "task_id": rank,
            "num_tasks": count,
            "scoring": scoring,
            "records": records,
        }
        (root / f"{split}_{policy}_part_{rank}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def write_eval(path: Path, schema_path: Path, flipped: set[str] | None = None) -> None:
    flipped = flipped or set()
    schema = load_schema_contract(schema_path)
    records = []
    for row in schema.rows:
        no_target = row["meta"]["no_target"]
        explicit_null = bool(no_target)
        if row["id"] in flipped:
            explicit_null = not explicit_null
        records.append(
            {
                "id": row["id"],
                "truth_exists": not no_target,
                "explicit_null": explicit_null,
            }
        )
    path.write_text(json.dumps({"refseg_overall": {"records": records}}), encoding="utf-8")


def make_formal_adapter(output_root: Path, run: str = "run", status: str = "finished") -> Path:
    run_dir = output_root / run
    adapter = run_dir / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (run_dir / "metrics.json").write_text(
        json.dumps({"status": status, "steps_completed": 4}), encoding="utf-8"
    )
    digest = "a" * 64
    (run_dir / "provenance_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_checkpoint": "/checkpoints/SAMTok/base",
                "config": {"sha256": digest},
                "source": {"sha256": digest},
                "data": {"sha256": digest, "row_ids_sha256": digest, "row_ids": ["id"]},
            }
        ),
        encoding="utf-8",
    )
    return adapter


def test_strict_shard_schema_eval_and_source_disjoint_contracts(tmp_path):
    train_path = tmp_path / "train.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    write_schema(train_path, "train")
    write_schema(holdout_path, "holdout")
    margins = [-1.0 if index % 2 == 0 else 1.0 for index in range(512)]
    for split, schema_path in (("train", train_path), ("holdout", holdout_path)):
        for policy in ("continued", "candidate"):
            write_margin_shards(tmp_path, split, policy, schema_path, margins)
    records, schema, _ = validate_margin_shards(tmp_path, "train", "continued", 2)
    assert len(records) == 512
    assert_disjoint_source_images(schema, load_schema_contract(holdout_path))

    eval_path = tmp_path / "eval.json"
    write_eval(eval_path, train_path)
    assert len(load_aligned_eval(eval_path, schema)) == 512

    bad_payload_path = tmp_path / "train_continued_part_0.json"
    payload = json.loads(bad_payload_path.read_text(encoding="utf-8"))
    payload["records"][0]["margin"] = float("nan")
    bad_payload_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        validate_margin_shards(tmp_path, "train", "continued", 2)

    payload["records"][0]["margin"] = payload["records"][0]["null_action_log_prob"]
    payload["schema_sha256"] = "0" * 64
    bad_payload_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema SHA mismatch"):
        validate_margin_shards(tmp_path, "train", "continued", 2)

    holdout_rows = write_schema(holdout_path, "holdout")
    holdout_rows[0]["meta"]["source_image_id"] = "train-0-positive"
    holdout_path.write_text(
        "".join(json.dumps(row) + "\n" for row in holdout_rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_source_images(schema, load_schema_contract(holdout_path))


def test_eval_accepts_sharded_order_but_requires_exact_ids_and_labels(tmp_path):
    schema_path = tmp_path / "schema.jsonl"
    write_schema(schema_path, "eval")
    schema = load_schema_contract(schema_path)
    eval_path = tmp_path / "eval.json"
    write_eval(eval_path, schema_path)
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    payload["refseg_overall"]["records"][0], payload["refseg_overall"]["records"][1] = (
        payload["refseg_overall"]["records"][1],
        payload["refseg_overall"]["records"][0],
    )
    eval_path.write_text(json.dumps(payload), encoding="utf-8")
    aligned = load_aligned_eval(eval_path, schema)
    assert list(aligned) == [str(row["id"]) for row in schema.rows]

    payload["refseg_overall"]["records"][0]["id"] = "unknown-id"
    eval_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly align"):
        load_aligned_eval(eval_path, schema)


def test_flip_prediction_requires_threshold_crossing_on_both_sides(tmp_path, monkeypatch):
    from tools import analyze_action_margin

    train_path = tmp_path / "train.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    write_schema(train_path, "train")
    holdout_rows = write_schema(holdout_path, "holdout")
    train_margins = [-1.0 if index % 2 == 0 else 1.0 for index in range(512)]
    continued_holdout = list(train_margins)
    candidate_holdout = list(train_margins)
    flipped_ids = {holdout_rows[index]["id"] for index in range(1, 12, 2)}
    for index in range(1, 10, 2):
        candidate_holdout[index] = -1.0
    continued_holdout[11] = 0.5
    candidate_holdout[11] = -1.0

    for policy in ("continued", "candidate"):
        write_margin_shards(tmp_path, "train", policy, train_path, train_margins)
    write_margin_shards(tmp_path, "holdout", "continued", holdout_path, continued_holdout)
    write_margin_shards(tmp_path, "holdout", "candidate", holdout_path, candidate_holdout)
    continued_eval = tmp_path / "continued_eval.json"
    candidate_eval = tmp_path / "candidate_eval.json"
    write_eval(continued_eval, holdout_path)
    write_eval(candidate_eval, holdout_path, flipped=flipped_ids)
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_action_margin.py",
            "--margin-root",
            str(tmp_path),
            "--num-shards",
            "2",
            "--continued-eval",
            str(continued_eval),
            "--candidate-eval",
            str(candidate_eval),
            "--permutations",
            "20",
            "--output",
            str(output),
        ],
    )
    analyze_action_margin.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["train_fitted_threshold"] == 1.0
    assert result["num_candidate_only_null_flips"] == 6
    assert result["flips_predicted_by_train_threshold"] == 5
    assert holdout_rows[11]["id"] not in result["predicted_flip_ids"]


def test_formal_adapter_requires_finished_standalone_run(tmp_path):
    output_root = tmp_path / "outputs" / "samtok_selective"
    adapter = make_formal_adapter(output_root)
    result = validate_formal_adapter(adapter, output_root)
    assert result["adapter"] == str(adapter.resolve())
    assert len(result["provenance_sha256"]) == 64

    unfinished = make_formal_adapter(output_root, run="unfinished", status="running")
    with pytest.raises(ValueError, match="not finished"):
        validate_formal_adapter(unfinished, output_root)
    external = make_formal_adapter(tmp_path / "PixVL" / "outputs", run="legacy")
    with pytest.raises(ValueError, match="direct formal run"):
        validate_formal_adapter(external, output_root)


def test_submit_uses_registered_tags_below_eight(tmp_path):
    faro_root = tmp_path / "FARO"
    (faro_root / "tools").mkdir(parents=True)
    os.symlink(
        REPO_ROOT / "tools" / "action_margin_contract.py",
        faro_root / "tools" / "action_margin_contract.py",
    )
    output_root = faro_root / "outputs" / "samtok_selective"
    continued = make_formal_adapter(output_root, "continued")
    candidate = make_formal_adapter(output_root, "candidate")
    train_schema = faro_root / "train.jsonl"
    holdout_schema = faro_root / "holdout.jsonl"
    write_schema(train_schema, "train")
    write_schema(holdout_schema, "holdout")
    (faro_root / "rjob_tags.txt").write_text("required-tag\n", encoding="utf-8")
    (faro_root / "rjob_tags_16gpu_partition.txt").write_text("wrong-tag\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "rjob_args.txt"
    fake_rjob = bin_dir / "rjob"
    fake_rjob.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n", encoding="utf-8")
    fake_rjob.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "FARO_ROOT": str(faro_root),
        "MODEL": str(model),
        "CONTINUED": str(continued),
        "CANDIDATE": str(candidate),
        "TRAIN_SCHEMA": str(train_schema),
        "HOLDOUT_SCHEMA": str(holdout_schema),
        "GPU_COUNT": "4",
        "TAGS_FILE": str(faro_root / "rjob_tags_16gpu_partition.txt"),
    }
    subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "submit_action_margin_diag.sh")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    arguments = capture.read_text(encoding="utf-8")
    assert "--positive-tags=required-tag" in arguments
    assert "wrong-tag" not in arguments
    assert "--gpu=4" in arguments
