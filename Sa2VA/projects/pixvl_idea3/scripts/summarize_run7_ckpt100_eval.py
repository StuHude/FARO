from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/run7_ckpt100_eval")

ref = json.loads((ROOT / "refcoco_grefcoco" / "refcoco_metrics.json").read_text())
gref = json.loads((ROOT / "refcoco_grefcoco" / "grefcoco_metrics.json").read_text())
split_summary = json.loads((ROOT / "split" / "summary.json").read_text())
split_relation = json.loads((ROOT / "split" / "relation.json").read_text())
split_geometry = json.loads((ROOT / "split" / "geometry.json").read_text())
split_refseg_overall = json.loads((ROOT / "split" / "refseg_overall.json").read_text())
split_semantic = json.loads((ROOT / "split" / "semantic.json").read_text())
split_dlc_reward = json.loads((ROOT / "split" / "dlc_reward.json").read_text())
dlc = json.loads((ROOT / "dlc_official" / "eval.json").read_text())

payload = {
    "refcoco": ref,
    "grefcoco": gref,
    "split": {
        "semantic": split_semantic,
        "relation": split_relation,
        "geometry": split_geometry,
        "refseg_overall": split_refseg_overall,
        "dlc_reward": split_dlc_reward,
        "summary": split_summary,
    },
    "dlc_official": {
        "avg_pos": dlc["avg_pos"],
        "avg_neg": dlc["avg_neg"],
        "avg": dlc["avg"],
    },
}

out = ROOT / "summary.json"
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
