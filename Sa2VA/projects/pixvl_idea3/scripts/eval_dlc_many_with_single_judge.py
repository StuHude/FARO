#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import inflect
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_official_module():
    mod_path = Path("/tmp/describe-anything/evaluation/eval_model_outputs.py")
    spec = importlib.util.spec_from_file_location("describe_anything_eval_model_outputs", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official scorer from {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--qa", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dlc_bench/qa.json")
    parser.add_argument("--class-names", dest="class_names", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dlc_bench/class_names.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--item", action="append", required=True, help="name=/abs/path/to/pred.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    official = load_official_module()
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype="auto",
    ).to(args.device).eval()

    def local_query(prompt: str, temperature: float, max_tokens: int, model_name: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                temperature=temperature,
                max_new_tokens=max_tokens,
                top_p=1.0,
                use_cache=True,
            )
        trimmed = output_ids[:, inputs.input_ids.shape[1]:]
        return tokenizer.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    official.query = local_query
    official.api_call_count = 0
    official.args = argparse.Namespace(api_call_limit=10**9)
    official.p = inflect.engine()

    with open(args.qa) as f:
        data_qa = json.load(f)
    with open(args.class_names) as f:
        data_class_names = json.load(f)
    p = inflect.engine()

    for item_spec in args.item:
        name, pred_path = item_spec.split("=", 1)
        pred_path = pred_path.strip()
        with open(pred_path) as f:
            data_pred = json.load(f)

        scores = {}
        scores_pos = {}
        scores_neg = {}
        eval_results = {}

        for key in data_qa.keys():
            key = str(key)
            pred_value = data_pred[key]
            class_name = data_class_names[key]

            if official.is_plural(class_name):
                recognition_question = f"Is it likely that the objects in the description are {class_name} or objects of a similar type? Again, It does not have to be an exact match."
            else:
                recognition_question = f"Is it likely that the object in the description is {p.a(class_name)} or an object of a similar type? Again, It does not have to be an exact match."
            recognition_question_dict = {
                "question": recognition_question,
                "choices": [("Yes", "correct"), ("No", "incorrect")],
                "type": "recognition",
            }
            question_dicts = [recognition_question_dict, *data_qa[key]]
            info = official.evaluate(
                question_dicts=question_dicts,
                pred_caption=pred_value,
                model=args.judge_model,
                temperature=0.0,
                max_tokens=300,
                response_override=None,
                key=key,
                verbose=False,
            )
            scores[key] = info["score"]
            scores_pos[key] = info["score_pos"]
            scores_neg[key] = info["score_neg"]
            eval_results[key] = {"pred": pred_value, **info}

        avg_pos = sum(scores_pos.values()) / len(scores_pos)
        valid_negs = [item for item in scores_neg.values() if item is not None]
        avg_neg = sum(valid_negs) / len(valid_negs)
        payload = {
            "avg_pos": avg_pos,
            "avg_neg": avg_neg,
            "avg": (avg_pos + avg_neg) / 2,
            "details": eval_results,
        }
        out_path = out_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(json.dumps({"name": name, "avg_pos": avg_pos, "avg_neg": avg_neg, "avg": (avg_pos + avg_neg) / 2}), flush=True)


if __name__ == "__main__":
    main()
