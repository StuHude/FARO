"""Build a three-slice NC-FEPO verifier manifest.

The cross-image slice is intentionally marked provisional: it is a hard
negative for the prompt/image pairing, not an official segmentation label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--positive", required=True)
    p.add_argument("--negative", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--per-slice", type=int, default=128)
    args = p.parse_args()
    positives = [r for r in read_rows(Path(args.positive)) if r.get("image_path") or r.get("image")]
    negatives = [r for r in read_rows(Path(args.negative)) if r.get("image_path") or r.get("image")]
    n = args.per_slice
    if len(positives) < n or len(negatives) < n:
        raise RuntimeError(f"need {n} rows per slice: positives={len(positives)} negatives={len(negatives)}")
    out = []
    for i, row in enumerate(positives[:n]):
        out.append({
            "id": f"positive-{i:04d}",
            "image_path": row.get("image_path") or row.get("image"),
            "prompt": row.get("prompt") or row.get("query"),
            "class": "positive",
            "source": row.get("source", "refseg_positive"),
        })
    for i, row in enumerate(negatives[:n]):
        out.append({
            "id": f"gres-negative-{i:04d}",
            "image_path": row.get("image_path") or row.get("image"),
            "prompt": row.get("prompt") or row.get("query"),
            "class": "negative",
            "source": "gres_no_target",
        })
    # Pair each positive prompt with a different positive image. This tests
    # image conditioning while remaining explicitly provisional.
    for i, row in enumerate(positives[:n]):
        other = positives[(i + max(1, n // 2)) % n]
        out.append({
            "id": f"cross-image-hard-{i:04d}",
            "image_path": other.get("image_path") or other.get("image"),
            "prompt": row.get("prompt") or row.get("query"),
            "class": "negative",
            "source": "cross_image_provisional",
            "paired_positive_id": f"positive-{i:04d}",
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
    print(f"wrote {output} positive={n} gres_negative={n} cross_image_provisional={n}")


if __name__ == "__main__":
    main()
