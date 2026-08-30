"""Build ordinary RL schemas from labeled SAMTok RefCOCO rows.

The input rows are used only for their RefCOCO image, query, and supervised
SAMTok mask tokens. Legacy track/cycle metadata is intentionally discarded.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MASK_RE = re.compile(r"<\|mt_\d{4}\|>")


def _answer_mask_tokens(answer: str) -> str | None:
    tokens = MASK_RE.findall(answer)
    if len(tokens) < 2:
        return None
    return "<|mt_start|>" + "".join(tokens) + "<|mt_end|>"


def build_refseg(path: Path, limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if len(rows) >= limit or not line.strip():
            continue
        item = json.loads(line)
        image = item.get("image")
        conversations = item.get("conversations") or []
        if not image or len(conversations) < 2:
            continue
        image = str(image)
        if image in seen:
            continue
        prompt = str(conversations[0].get("value", "")).replace("<image>", "").strip()
        query = re.sub(r"^please\s+segment\s+", "", prompt, flags=re.IGNORECASE).strip()
        mask_tokens = _answer_mask_tokens(str(conversations[1].get("value", "")))
        if not query or not mask_tokens:
            continue
        seen.add(image)
        rows.append({
            "id": f"refseg-samtok-{len(rows):05d}",
            "task": "refseg",
            "source": "refcoco_labeled_samtok",
            "split": "train",
            "image_path": image,
            "query": query,
            "mask_tokens": mask_tokens,
        })
    return rows


def build_existence(positive: list[dict[str, object]], negative_path: Path, limit: int) -> list[dict[str, object]]:
    negatives: list[dict[str, str]] = []
    seen_neg: set[str] = set()
    for line in negative_path.read_text(encoding="utf-8").splitlines():
        if len(negatives) >= limit or not line.strip():
            continue
        item = json.loads(line)
        image = item.get("image") or item.get("image_path")
        prompt = item.get("prompt") or item.get("query")
        if not image or not prompt:
            continue
        image = str(image)
        image_key = Path(image).name
        if image_key in seen_neg:
            continue
        prompt = re.sub(r"^\s*please\s+segment\s+", "", str(prompt), flags=re.IGNORECASE)
        prompt = re.split(r"\s*\bif\s+no\s+described\s+target\s+exists\b", prompt, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if not prompt:
            continue
        seen_neg.add(image_key)
        negatives.append({"image_path": image, "query": prompt})

    # Select one image-disjoint positive per negative with a nearby prompt
    # length. This removes source/path leakage and the original class skew.
    candidates = [item for item in positive if Path(str(item["image_path"])).name not in seen_neg]
    selected: list[dict[str, object]] = []
    for negative in sorted(negatives, key=lambda item: len(item["query"].split())):
        if not candidates:
            break
        target_len = len(negative["query"].split())
        best = min(range(len(candidates)), key=lambda idx: abs(len(str(candidates[idx]["query"]).split()) - target_len))
        selected.append(candidates.pop(best))
    negatives = negatives[: len(selected)]

    rows: list[dict[str, object]] = []
    for idx, item in enumerate(selected):
        rows.append({"id": f"existence-refcoco-positive-{idx:05d}", "task": "existence", "source": "refcoco_labeled_positive", "split": "train", "image_path": item["image_path"], "query": item["query"], "answer": "Target exists."})
    for idx, item in enumerate(negatives):
        rows.append({"id": f"existence-gres-negative-{idx:05d}", "task": "existence", "source": "gres_labeled_negative", "split": "train", "image_path": item["image_path"], "query": item["query"], "answer": "No target."})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--refseg-output", required=True)
    parser.add_argument("--existence-output", required=True)
    parser.add_argument("--limit", type=int, default=1024)
    args = parser.parse_args()
    refseg = build_refseg(Path(args.input), args.limit)
    existence = build_existence(refseg, Path(args.negative), min(args.limit, len(refseg)))
    for output, rows in ((args.refseg_output, refseg), (args.existence_output, existence)):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        print(f"wrote {path} rows={len(rows)}")


if __name__ == "__main__":
    main()
