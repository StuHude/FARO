import json
from pathlib import Path

from tools.build_nc_fepo_manifest import main


def test_manifest_has_three_slices(tmp_path, monkeypatch):
    pos = tmp_path / "p.jsonl"
    neg = tmp_path / "n.jsonl"
    pos.write_text("\n".join(json.dumps({"image": f"p{i}.jpg", "prompt": f"p{i}"}) for i in range(4)))
    neg.write_text("\n".join(json.dumps({"image": f"n{i}.jpg", "prompt": f"n{i}"}) for i in range(4)))
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr("sys.argv", ["x", "--positive", str(pos), "--negative", str(neg), "--output", str(out), "--per-slice", "2"])
    main()
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [sum(r["class"] == c for r in rows) for c in ("positive", "negative")] == [2, 4]
    assert sum(r["source"] == "cross_image_provisional" for r in rows) == 2
