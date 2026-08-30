from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_IMPORT_PREFIXES = ("projects." + "pixvl_",)
FORBIDDEN_PATH_FRAGMENTS = (
    "Pix" + "VL_ailab",
    "third_party/" + "Sa2VA",
)


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


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def source_tree_hash(root: str | Path) -> tuple[str, dict[str, str]]:
    root = Path(root).resolve()
    entries: dict[str, str] = {}
    for path in _python_files(root):
        entries[path.relative_to(root).as_posix()] = sha256_file(path)
    combined = sha256_text_items(f"{key}:{value}" for key, value in entries.items())
    return combined, entries


def assert_training_source_clean(root: str | Path) -> None:
    root = Path(root).resolve()
    violations: list[str] = []
    for path in _python_files(root):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    violations.append(f"{path}: forbidden import {name}")
        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            if fragment in source:
                violations.append(f"{path}: forbidden source path fragment")
    if violations:
        raise RuntimeError("Standalone source guard failed:\n" + "\n".join(violations))


def guard_runtime_environment() -> None:
    entries = list(sys.path)
    entries.extend(os.environ.get("PYTHONPATH", "").split(os.pathsep))
    violations = [entry for entry in entries if entry and any(part in entry for part in FORBIDDEN_PATH_FRAGMENTS)]
    if violations:
        raise RuntimeError("Forbidden training runtime paths: " + ", ".join(sorted(set(violations))))


def validate_base_checkpoint(path: str | Path) -> Path:
    path = Path(path).resolve()
    required = (
        "config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "sam2.1_hiera_large.pt",
        "mask_tokenizer_256x2.pth",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Base checkpoint is missing required SAMTok artifacts: {missing}")
    if (path / "adapter_config.json").exists():
        raise ValueError("Base checkpoint must not be an adapter or derived training output")
    payload = json.loads((path / "config.json").read_text(encoding="utf-8"))
    architectures = payload.get("architectures") or []
    if "Qwen3VLForConditionalGeneration" not in architectures:
        raise ValueError(f"Unexpected base architecture: {architectures}")
    return path


def validate_declared_paths(config: dict[str, Any], repo_root: str | Path) -> None:
    repo_root = Path(repo_root).resolve()
    output_root = repo_root / "outputs" / "samtok_selective"
    cache_root = repo_root / "data" / "samtok_selective_cache"
    output_dir = Path(config["checkpoint"]["output_dir"]).resolve()
    cache_path = Path(config["data"]["cache_path"]).resolve()
    if output_root not in output_dir.parents:
        raise ValueError(f"Training output must be below {output_root}")
    if cache_root not in cache_path.parents:
        raise ValueError(f"Training cache must be below {cache_root}")


def _checkpoint_files(root: Path) -> list[Path]:
    names = {
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "sam2.1_hiera_large.pt",
        "mask_tokenizer_256x2.pth",
    }
    files = [root / name for name in sorted(names) if (root / name).is_file()]
    files.extend(sorted(root.glob("model-*-of-*.safetensors")))
    return files


def build_manifest(config: dict[str, Any], config_path: str | Path, package_root: str | Path) -> dict[str, Any]:
    base = validate_base_checkpoint(config["model"]["base_checkpoint"])
    repo_root = Path(package_root).resolve().parents[2]
    data_path = Path(config["data"]["jsonl"]).resolve()
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    row_ids = [str(row["id"]) for row in rows]
    source_hash, source_files = source_tree_hash(package_root)
    codec_runtime = Path(package_root).resolve().parents[0] / "samtok" / "demo" / "gradio" / "sam2.py"
    artifacts = {path.relative_to(base).as_posix(): sha256_file(path) for path in _checkpoint_files(base)}
    external_runtime_files = [repo_root / "third_party" / "TRANSFORMERS_RUNTIME.lock"]
    external_runtime_files.extend(sorted((repo_root / "vendor" / "wheels").glob("*.whl")))
    adapter_init = config["checkpoint"].get("adapter_init")
    initialization = None
    if adapter_init:
        adapter_root = Path(adapter_init).resolve()
        standalone_output_root = (repo_root / "outputs" / "samtok_selective").resolve()
        if standalone_output_root not in adapter_root.parents:
            raise ValueError("Adapter initialization must stay within standalone outputs")
        files = {
            name: sha256_file(adapter_root / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")
            if (adapter_root / name).is_file()
        }
        if len(files) != 2:
            raise ValueError(f"Adapter initialization is incomplete: {adapter_root}")
        initialization = {"path": str(adapter_root), "files": files}
    return {
        "schema_version": 1,
        "base_checkpoint": str(base),
        "initialization_adapter": initialization,
        "artifacts": artifacts,
        "config": {"path": str(Path(config_path).resolve()), "sha256": sha256_file(config_path)},
        "source": {"root": str(Path(package_root).resolve()), "sha256": source_hash, "files": source_files},
        "codec_runtime": {"path": str(codec_runtime), "sha256": sha256_file(codec_runtime)},
        "external_runtime": {
            str(path.relative_to(repo_root)): sha256_file(path)
            for path in external_runtime_files
        },
        "data": {
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "row_count": len(rows),
            "row_ids_sha256": sha256_text_items(row_ids),
            "row_ids": row_ids,
        },
    }


def runtime_module_files() -> dict[str, str | None]:
    modules: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "accelerate", "projects.samtok.demo.gradio.sam2"):
        module = importlib.import_module(name)
        modules[name] = getattr(module, "__file__", None)
    prefixes = ("transformers", "peft", "accelerate", "projects.samtok")
    for name, module in sorted(sys.modules.items()):
        if name.startswith(prefixes) and module is not None:
            path = getattr(module, "__file__", None)
            if path:
                modules[name] = path
    violations = {
        name: path
        for name, path in modules.items()
        if path and any(fragment in path for fragment in FORBIDDEN_PATH_FRAGMENTS)
    }
    if violations:
        raise RuntimeError(f"Runtime modules resolved through forbidden paths: {violations}")
    return modules


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("guard",))
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    assert_training_source_clean(args.root)
    guard_runtime_environment()
    print(json.dumps({"status": "ok", "root": str(Path(args.root).resolve())}))


if __name__ == "__main__":
    main()
