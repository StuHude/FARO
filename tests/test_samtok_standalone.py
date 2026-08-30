import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from projects.samtok_selective.data import (
    PairedBatchSampler,
    SelectiveRefSegDataset,
    load_refseg_jsonl,
    validate_pair_parity,
)
from projects.samtok_selective.fepo_ampcpo_trainer import canonical_null_margin_terms
from projects.samtok_selective.geometry_reward import area_similarity, ciou
from projects.samtok_selective.manifests import (
    assert_training_source_clean,
    validate_base_checkpoint,
)
from projects.samtok_selective.mask_codec import PINNED_RUNTIME_SHA256, RUNTIME_PATH
from projects.samtok_selective.modeling import (
    answer_token_cross_entropy,
    normalize_qwen3vl_config_payload,
)
from projects.samtok_selective.selective_rl_trainer import accept_v7_step


class FakeCodec:
    def __init__(self):
        self.encoded = []

    def cache_key(self, image_path, mask_obj):
        return f"{image_path}:{mask_obj['counts']}"

    def encode_mask(self, image, mask_obj):
        self.encoded.append(mask_obj["counts"])
        return [7, 263]

    def codes_to_text(self, codes):
        return "<|mt_start|><|mt_0007|><|mt_0263|><|mt_end|>"


def _write_fixture(tmp_path: Path) -> tuple[Path, list[dict]]:
    image_paths = []
    for index in range(4):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (8, 8), (index, index, index)).save(path)
        image_paths.append(path)
    rows = []
    for pair_index in range(2):
        for no_target in (False, True):
            suffix = "negative" if no_target else "positive"
            rows.append(
                {
                    "id": f"row-{pair_index}-{suffix}",
                    "pair_id": f"pair-{pair_index}",
                    "task": "refseg",
                    "image_path": str(image_paths[len(rows)]),
                    "query": f"target {pair_index}",
                    "mask": {"format": "rle", "counts": f"rle-{len(rows)}", "size": [8, 8]},
                    "meta": {"no_target": no_target},
                }
            )
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path, rows


def test_pair_parity_and_sampler_keep_outcomes_together(tmp_path):
    path, rows = _write_fixture(tmp_path)
    assert validate_pair_parity(rows, expected_rows=4, expected_no_target_rows=2) == {
        "rows": 4,
        "pairs": 2,
        "no_target_rows": 2,
    }
    codec = FakeCodec()
    dataset = SelectiveRefSegDataset(
        path,
        'Segment "{query}".',
        codec,
        tmp_path / "cache.sqlite3",
        expected_rows=4,
        expected_no_target_rows=2,
    )
    batches = list(PairedBatchSampler(dataset, pairs_per_batch=1, seed=42))
    assert len(batches) == 2
    assert all([dataset.rows[index]["meta"]["no_target"] for index in batch] == [False, True] for batch in batches)


def test_no_target_never_uses_mask_codec(tmp_path):
    path, _ = _write_fixture(tmp_path)
    codec = FakeCodec()
    dataset = SelectiveRefSegDataset(path, "{query}", codec, tmp_path / "cache.sqlite3")
    negative_index = next(index for index, row in enumerate(dataset.rows) if row["meta"]["no_target"])
    positive_index = next(index for index, row in enumerate(dataset.rows) if not row["meta"]["no_target"])
    assert dataset[negative_index]["answer_text"] == "No target."
    assert codec.encoded == []
    assert dataset[positive_index]["codes"] == [7, 263]
    assert len(codec.encoded) == 1


def test_refseg_reader_rejects_non_refseg(tmp_path):
    path, rows = _write_fixture(tmp_path)
    rows[0]["task"] = "maskcap"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Only refseg"):
        load_refseg_jsonl(path)


def test_answer_ce_ignores_prompt_tokens():
    logits = torch.zeros(1, 3, 5)
    input_ids = torch.tensor([[0, 1, 2]])
    attention = torch.ones_like(input_ids)
    answer_mask = torch.tensor([[False, False, True]])
    logits[0, 1, 2] = 8.0
    loss = answer_token_cross_entropy(logits, input_ids, attention, answer_mask)
    assert loss.item() < 0.01


def test_answer_ce_weights_samples_not_answer_lengths():
    logits = torch.zeros(2, 5, 5)
    input_ids = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    attention = torch.ones_like(input_ids)
    answer_mask = torch.tensor(
        [[False, True, True, True, True], [False, False, False, False, True]]
    )
    # The long first answer is nearly perfect; the one-token second answer is
    # uniform. Equal sample weighting therefore averages their two losses.
    for position, target in enumerate(input_ids[0, 1:].tolist()):
        logits[0, position, target] = 8.0
    loss = answer_token_cross_entropy(logits, input_ids, attention, answer_mask)
    expected = 0.5 * torch.log(torch.tensor(5.0))
    assert torch.isclose(loss, expected, atol=0.01)


def test_ampcpo_margin_uses_first_null_action_not_phrase_length():
    class Tokenizer:
        im_end_token_id = 99

        def convert_tokens_to_ids(self, token):
            assert token == "<|im_end|>"
            return self.im_end_token_id

        def decode(self, token_ids, skip_special_tokens=False):
            assert token_ids == [7, 8, 9]
            return "No target."

    input_ids = torch.tensor([[0, 1, 7, 8, 9, 99]])
    answer_mask = torch.tensor([[False, False, True, True, True, True]])
    log_probs = torch.zeros(1, 6, 16)
    # At the answer boundary, null's first token beats the mask action by 2.
    log_probs[0, 1, 7] = 3.0
    log_probs[0, 1, 4] = 1.0
    # Later phrase tokens must not add to the action margin.
    log_probs[0, 2, 8] = -40.0
    log_probs[0, 3, 9] = -40.0
    margins = canonical_null_margin_terms(
        log_probs,
        input_ids,
        answer_mask,
        torch.tensor([True]),
        Tokenizer(),
        [4],
    )
    assert torch.equal(margins, torch.tensor([2.0]))


def test_v5_dev_rope_parameters_are_preserved_for_pinned_runtime():
    source = {
        "text_config": {
            "rope_scaling": None,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
            },
        }
    }
    normalized = normalize_qwen3vl_config_payload(source)
    assert source["text_config"]["rope_scaling"] is None
    assert "rope_parameters" not in normalized["text_config"]
    assert normalized["text_config"]["rope_scaling"]["mrope_section"] == [24, 20, 20]


def test_geometry_metrics_have_defined_empty_behavior():
    empty = np.zeros((4, 4), dtype=np.uint8)
    full = np.ones((4, 4), dtype=np.uint8)
    assert ciou(empty, empty) == 1.0
    assert ciou(empty, full) == 0.0
    assert area_similarity(empty, full) == 0.0


def test_source_guard_and_pinned_runtime():
    package = Path(__file__).resolve().parents[1] / "Sa2VA" / "projects" / "samtok_selective"
    assert_training_source_clean(package)
    import hashlib

    assert hashlib.sha256(RUNTIME_PATH.read_bytes()).hexdigest() == PINNED_RUNTIME_SHA256


def test_base_checkpoint_guard_rejects_adapter(tmp_path):
    required = (
        "config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "sam2.1_hiera_large.pt",
        "mask_tokenizer_256x2.pth",
    )
    for name in required:
        payload = {"architectures": ["Qwen3VLForConditionalGeneration"]} if name == "config.json" else {}
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    assert validate_base_checkpoint(tmp_path) == tmp_path.resolve()
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be an adapter"):
        validate_base_checkpoint(tmp_path)


def test_v7_decision_rejects_either_budget_violation():
    assert accept_v7_step(null_risk=0.9, null_budget=1.0, anchor_kl=0.1, kl_budget=0.2).accepted
    assert not accept_v7_step(null_risk=1.1, null_budget=1.0, anchor_kl=0.1, kl_budget=0.2).accepted
    assert not accept_v7_step(null_risk=0.9, null_budget=1.0, anchor_kl=0.3, kl_budget=0.2).accepted


def test_submit_script_has_dnacoding_contract():
    script = Path(__file__).resolve().parents[1] / "scripts" / "submit_samtok_standalone_sft.sh"
    text = script.read_text(encoding="utf-8")
    assert "--namespace=ailab-dnacoding" in text
    assert "--positive-tags=\"$POSITIVE_TAGS\"" in text
    assert "JOB_NAME must start with dna-" in text
    assert "--nproc_per_node=2" in text


def test_standalone_eval_requires_full_paired_holdout_and_adaptive_fallback():
    script = Path(__file__).resolve().parents[1] / "scripts" / "submit_samtok_standalone_eval_adaptive.sh"
    text = script.read_text(encoding="utf-8")
    assert "grefcoco_selective_holdout_256.jsonl" in text
    assert "-eq 512" in text
    assert "submit_fepo_eval_adaptive.sh" in text
    assert "outputs/samtok_selective" in text


def test_adaptive_eval_uses_approved_positive_tags_at_every_gpu_level():
    script = Path(__file__).resolve().parents[1] / "scripts" / "submit_fepo_eval_sharded.sh"
    text = script.read_text(encoding="utf-8")
    assert "DEFAULT_TAGS_FILE=$FARO_ROOT/rjob_tags.txt" in text
    assert "rjob_tags_16gpu_partition.txt" not in text
    assert '--positive-tags="$POSITIVE_TAGS"' in text
