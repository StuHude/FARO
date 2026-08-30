"""Convert standard supervised RefCOCO SAMTok rows into the direct SFT schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=1024)
    args = p.parse_args()
    rows = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip() or len(rows) >= args.limit:
            continue
        row = json.loads(line); conversations = row.get("conversations") or []
        if len(conversations) < 2 or not row.get("image"):
            continue
        prompt = str(conversations[0].get("value", "")).replace("<image>", "").strip()
        answer = str(conversations[1].get("value", "")).strip()
        rows.append({"id": f"refcoco-direct-{len(rows):05d}", "task": "direct_sft", "source": "refcoco_supervised", "split": "train", "image_path": row["image"], "query": prompt, "answer": answer})
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
