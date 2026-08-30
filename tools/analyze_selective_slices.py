"""Paired cIoU/boundary slice diagnostics for complete selective holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.bootstrap_metrics import bootstrap_paired_delta


def _records(path: Path, section: str) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload[section]["records"]
    return {str(row["id"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--section", default="refseg_overall")
    parser.add_argument("--repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-drop", type=float, default=0.01)
    parser.add_argument("--min-noninferior-slices", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base, candidate = _records(args.base, args.section), _records(args.candidate, args.section)
    ids = sorted(set(base) & set(candidate))
    if len(ids) != 512 or len(ids) != len(base) or len(ids) != len(candidate):
        raise ValueError(f"slice analysis requires exactly 512 paired rows, got base={len(base)} candidate={len(candidate)} paired={len(ids)}")
    if any("boundary_iou" not in candidate[row_id] for row_id in ids):
        raise ValueError("candidate records lack boundary_iou; rerun evaluator with diagnostics enabled")
    if any("boundary_iou" not in base[row_id] for row_id in ids):
        raise ValueError("base records lack boundary_iou; rerun the baseline evaluator before slice gating")

    names = ("small", "thin", "boundary_hard", "area_stratum:small", "area_stratum:medium", "area_stratum:large")
    report = {"num_paired": len(ids), "slices": {}}
    noninferior = 0
    for index, name in enumerate(names):
        def selected(row: dict) -> bool:
            metadata = row.get("slice_metadata") or {}
            if name.startswith("area_stratum:"):
                return metadata.get("area_stratum") == name.split(":", 1)[1]
            return bool(metadata.get(name, False))

        slice_ids = [row_id for row_id in ids if selected(candidate[row_id])]
        if not slice_ids:
            continue
        ciou = bootstrap_paired_delta(
            [base[row_id]["ciou"] for row_id in slice_ids],
            [candidate[row_id]["ciou"] for row_id in slice_ids],
            repeats=args.repeats,
            seed=args.seed + index,
        )
        boundary = bootstrap_paired_delta(
            [base[row_id]["boundary_iou"] for row_id in slice_ids],
            [candidate[row_id]["boundary_iou"] for row_id in slice_ids],
            repeats=args.repeats,
            seed=args.seed + 100 + index,
        )
        passed = ciou["ci95"][0] >= -args.max_drop and boundary["ci95"][0] >= -args.max_drop
        noninferior += int(passed)
        report["slices"][name] = {
            "n": len(slice_ids),
            "ciou_delta": ciou,
            "boundary_iou_delta": boundary,
            "noninferior": passed,
        }
    report["noninferior_slice_count"] = noninferior
    report["slice_gate"] = noninferior >= args.min_noninferior_slices
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
