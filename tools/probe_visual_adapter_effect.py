from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from projects.pixvl_idea1.datasets import UnifiedRegionDataset
from projects.pixvl_idea1.trainers.common import (
    build_model_bundle,
    build_prompt_and_answer_ids,
    load_config,
    move_inputs_to_device,
)
from projects.pixvl_idea3.eval.eval_mvp_bundle import build_eval_model_bundle


def active(model):
    value = getattr(model, "active_adapters", None)
    return list(value() if callable(value) else value or [])


def set_adapter(model, names):
    tuner = getattr(model, "base_model", None)
    if tuner is None or not hasattr(tuner, "set_adapter"):
        raise RuntimeError("missing PEFT tuner set_adapter")
    tuner.set_adapter(names)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--anchor-adapter", required=True)
    parser.add_argument("--visual-adapter", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _logit_probe(model, processor, sample, label):
    prompt_inputs, _ = build_prompt_and_answer_ids(
        processor,
        sample["image"],
        sample["prompt_text"],
        sample["answer_text"],
    )
    prompt_inputs = move_inputs_to_device(prompt_inputs, next(model.parameters()).device)
    with torch.inference_mode():
        logits = model(**prompt_inputs, use_cache=False).logits[:, -1, :].float()
    values, indices = torch.topk(logits, k=8, dim=-1)
    return {
        "label": label,
        "active_adapters": active(model),
        "max_logit": float(logits.max().item()),
        "min_logit": float(logits.min().item()),
        "top8_token_ids": indices[0].tolist(),
        "top8_logits": [float(x) for x in values[0].tolist()],
        "logits": logits.cpu(),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    schema = Path(args.schema)
    dataset = UnifiedRegionDataset(
        schema_files=[str(schema)],
        model_name_or_path=cfg["model"]["processor_name_or_path"],
        mask_tokenizer_path=cfg["model"]["mask_tokenizer_path"],
        sam2_ckpt_path=cfg["model"]["sam2_ckpt_path"],
        cache_path=cfg["paths"]["mask_code_cache"],
        task_mix={"refseg": 1.0},
        source_mix=None,
        prompt_templates=cfg["data"]["prompts"],
        overlay_cfg=cfg["data"]["overlay"],
    )
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"index {args.index} outside schema with {len(dataset)} rows")
    sample = dataset[args.index]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    anchor_model, processor = build_model_bundle(
        cfg, trainable=False, adapter_path=args.anchor_adapter
    )
    anchor_model.to(device).eval()
    anchor_probe = _logit_probe(anchor_model, processor, sample, "anchor_only")
    anchor_logits = anchor_probe.pop("logits")
    del anchor_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    combined_model, processor = build_eval_model_bundle(
        cfg,
        adapter_path=args.visual_adapter,
        anchor_adapter_path=args.anchor_adapter,
    )
    combined_model.to(device).eval()
    combined_probe = _logit_probe(combined_model, processor, sample, "combined")
    combined_logits = combined_probe.pop("logits")

    set_adapter(combined_model, "anchor")
    combined_anchor_probe = _logit_probe(
        combined_model, processor, sample, "combined_anchor_only"
    )
    combined_anchor_logits = combined_anchor_probe.pop("logits")

    set_adapter(combined_model, "visual")
    combined_visual_probe = _logit_probe(
        combined_model, processor, sample, "combined_visual_only"
    )
    combined_visual_logits = combined_visual_probe.pop("logits")

    merged_probe = None
    merged_logits = None
    merge_error = None
    if hasattr(combined_model, "add_weighted_adapter"):
        try:
            combined_model.add_weighted_adapter(
                ["anchor", "visual"],
                [1.0, 1.0],
                adapter_name="merged",
                combination_type="svd",
                svd_rank=128,
            )
            set_adapter(combined_model, "merged")
            merged_probe = _logit_probe(combined_model, processor, sample, "merged")
            merged_logits = merged_probe.pop("logits")
        except Exception as exc:  # pragma: no cover - depends on PEFT runtime
            merge_error = f"{type(exc).__name__}: {exc}"

    def diff(lhs, rhs):
        delta = (lhs - rhs).abs()
        return {
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
            "changed_logits": int((delta > 1e-6).sum().item()),
        }

    payload = {
        "schema": str(schema),
        "index": args.index,
        "sample_id": sample.get("id"),
        "truth_exists": not bool((sample.get("meta") or {}).get("no_target", False)),
        "anchor": anchor_probe,
        "combined": combined_probe,
        "combined_anchor_only": combined_anchor_probe,
        "combined_visual_only": combined_visual_probe,
        "merged": merged_probe,
        "merge_error": merge_error,
        "diffs": {
            "combined_vs_anchor": diff(combined_logits, anchor_logits),
            "combined_anchor_vs_anchor": diff(combined_anchor_logits, anchor_logits),
            "combined_visual_vs_anchor": diff(combined_visual_logits, anchor_logits),
            "combined_vs_combined_anchor": diff(combined_logits, combined_anchor_logits),
            "merged_vs_anchor": diff(merged_logits, anchor_logits) if merged_logits is not None else None,
            "merged_vs_combined": diff(merged_logits, combined_logits) if merged_logits is not None else None,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
