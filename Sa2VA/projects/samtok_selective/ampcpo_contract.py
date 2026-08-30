from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, validate_config


STAGE = "fepo_ampcpo_smoke_2gpu"
EXPECTED_STEPS = 20


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def expected_continued_adapter(repo_root: str | Path = REPO_ROOT) -> Path:
    return Path(repo_root).resolve() / "outputs" / "samtok_selective" / "continued_sft" / "adapter"


def validate_ampcpo_config(config: dict[str, Any], repo_root: str | Path = REPO_ROOT) -> None:
    validate_config(config)
    repo_root = Path(repo_root).resolve()
    if config.get("stage") != STAGE:
        raise ValueError(f"AM-CPPO smoke stage must be {STAGE}")
    if int(config["optimizer"]["max_steps"]) != EXPECTED_STEPS:
        raise ValueError(f"AM-CPPO smoke must run exactly {EXPECTED_STEPS} steps")
    if int(config["runtime"]["expected_world_size"]) != 2:
        raise ValueError("AM-CPPO smoke requires exactly two processes")
    expected_adapter = expected_continued_adapter(repo_root)
    configured_adapter = Path(config["checkpoint"].get("adapter_init") or "").resolve()
    if configured_adapter != expected_adapter:
        raise ValueError(f"AM-CPPO must initialize from {expected_adapter}")
    expected_output = repo_root / "outputs" / "samtok_selective" / STAGE
    if Path(config["checkpoint"]["output_dir"]).resolve() != expected_output:
        raise ValueError(f"AM-CPPO output must be {expected_output}")

    method = config.get("ampcpo")
    if not isinstance(method, dict) or method.get("method") != "standalone_samtok_am_cppo":
        raise ValueError("Missing standalone_samtok_am_cppo method contract")
    if method.get("positive_reward") != "plain_ciou":
        raise ValueError("AM-CPPO positive reward must be plain_ciou")
    if method.get("negative_objective") != "canonical_no_target_ce":
        raise ValueError("AM-CPPO negative objective must be canonical_no_target_ce")
    if method.get("margin_constraint") != "first_null_token_vs_mask_start_hinge":
        raise ValueError("AM-CPPO margin constraint is not registered")
    clip = float(method.get("clip_epsilon", -1.0))
    if not 0.0 < clip < 1.0:
        raise ValueError("clip_epsilon must be in (0, 1)")
    for key in ("policy_weight", "null_ce_weight", "margin_weight"):
        value = float(method.get(key, -1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be finite and nonnegative")
    margin_target = float(method.get("margin_target", float("nan")))
    if not math.isfinite(margin_target):
        raise ValueError("margin_target must be finite")


def validate_continued_adapter(
    adapter_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    expected_path: str | Path | None = None,
    hash_model: bool = True,
) -> dict[str, Any]:
    adapter = Path(adapter_path).resolve()
    expected = (
        Path(expected_path).resolve()
        if expected_path is not None
        else expected_continued_adapter(repo_root)
    )
    if adapter != expected:
        raise ValueError(f"AM-CPPO adapter must be the formal continued-SFT adapter: {expected}")
    required = ("adapter_config.json", "adapter_model.safetensors")
    for filename in required:
        artifact = adapter / filename
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f"continued-SFT adapter is missing nonempty {filename}: {adapter}")

    run_dir = adapter.parent
    metrics_path = run_dir / "metrics.json"
    provenance_path = run_dir / "provenance_manifest.json"
    if not metrics_path.is_file() or not provenance_path.is_file():
        raise ValueError("continued-SFT requires metrics.json and provenance_manifest.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or metrics.get("status") != "finished":
        raise ValueError("continued-SFT metrics must have status=finished")
    if type(metrics.get("steps_completed")) is not int or metrics["steps_completed"] < 1:
        raise ValueError("continued-SFT metrics must record positive steps_completed")
    if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
        raise ValueError("continued-SFT provenance schema is invalid")
    base = provenance.get("base_checkpoint")
    if not isinstance(base, str) or "samtok" not in base.lower():
        raise ValueError("continued-SFT provenance does not identify a SAMTok base")
    return {
        "path": str(adapter),
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "adapter_model_sha256": (
            sha256_file(adapter / "adapter_model.safetensors") if hash_model else None
        ),
        "metrics_sha256": sha256_file(metrics_path),
        "provenance_sha256": sha256_file(provenance_path),
        "parent_stage": metrics.get("stage"),
        "parent_steps_completed": metrics["steps_completed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--skip-model-hash", action="store_true")
    args = parser.parse_args()
    result = validate_continued_adapter(
        args.adapter, repo_root=args.repo_root, hash_model=not args.skip_model_hash
    )
    print(json.dumps({"status": "ok", "initialization": result}, sort_keys=True))


if __name__ == "__main__":
    main()
