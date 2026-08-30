from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tools.score_samtok_action_margin import (  # noqa: E402
    action_log_probabilities,
    append_teacher_forcing_target,
    resolve_single_token_ids,
)


def test_scores_first_null_action_and_mask_candidate_set():
    logits = torch.zeros((1, 6, 8), dtype=torch.float32)
    logits[0, 2, 1] = 3.0
    logits[0, 3, 2] = 2.0
    logits[0, 4, 3] = 1.0
    logits[0, 2, 4] = 0.5
    logits[0, 2, 5] = 1.5
    null_log_prob, mask_log_prob, margin = action_log_probabilities(
        logits, prompt_length=3, null_token_ids=[1, 2, 3], mask_start_token_ids=[4, 5]
    )
    log_probs = torch.log_softmax(logits[0, 2], dim=-1)
    expected_null = log_probs[1]
    expected_mask = torch.logsumexp(log_probs[torch.tensor([4, 5])], dim=0)
    assert null_log_prob == pytest.approx(expected_null.item())
    assert mask_log_prob == pytest.approx(expected_mask.item())
    assert margin == pytest.approx((expected_null - expected_mask).item())

    inputs = {"input_ids": torch.tensor([[7, 6, 5]]), "attention_mask": torch.ones((1, 3))}
    extended, prompt_length = append_teacher_forcing_target(inputs, [1, 2, 3])
    assert prompt_length == 3
    assert extended["input_ids"].tolist() == [[7, 6, 5, 1, 2, 3]]
    assert inputs["input_ids"].tolist() == [[7, 6, 5]]


def test_configurable_mask_candidates_must_be_unique_single_tokens():
    class Tokenizer:
        unk_token_id = 99

        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return {"a": [1], "b": [2], "many": [1, 2], "unknown": [99]}[text]

    tokenizer = Tokenizer()
    assert resolve_single_token_ids(tokenizer, ["a", "b"]) == [1, 2]
    with pytest.raises(ValueError, match="one token"):
        resolve_single_token_ids(tokenizer, ["many"])
    with pytest.raises(ValueError, match="unique"):
        resolve_single_token_ids(tokenizer, ["a", "a"])
    with pytest.raises(ValueError, match="unavailable"):
        resolve_single_token_ids(tokenizer, ["unknown"])
