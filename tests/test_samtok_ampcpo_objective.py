from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from projects.samtok_selective.fepo_ampcpo_trainer import (  # noqa: E402
    canonical_null_margin_terms,
    clipped_policy_loss,
    per_sample_answer_nll,
)


def test_ampcpo_answer_nll_selects_only_answer_tokens():
    logits = torch.zeros(2, 4, 5)
    input_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    attention = torch.ones_like(input_ids)
    answer_mask = torch.tensor(
        [[False, False, True, True], [False, False, False, True]]
    )
    logits[0, 1, 2] = 8.0
    logits[0, 2, 3] = 8.0
    log_probs = torch.log_softmax(logits, dim=-1)
    losses = per_sample_answer_nll(log_probs, input_ids, attention, answer_mask)
    assert losses[0].item() < 0.01
    assert losses[1].item() == pytest.approx(torch.log(torch.tensor(5.0)).item())


def test_ampcpo_clipped_surrogate_clips_large_policy_ratio():
    current = torch.tensor([1.0], requires_grad=True)
    old = torch.tensor([0.0])
    advantages = torch.tensor([2.0])
    loss, clip_fraction = clipped_policy_loss(current, old, advantages, 0.2)
    assert loss.item() == pytest.approx(-2.4)
    assert clip_fraction.item() == 1.0


def test_ampcpo_margin_validates_phrase_but_compares_first_actions():
    class Tokenizer:
        unk_token_id = 99

        @staticmethod
        def convert_tokens_to_ids(token):
            assert token == "<|im_end|>"
            return 3

        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            assert not skip_special_tokens
            assert token_ids == [1, 2]
            return "No target."

    logits = torch.zeros((1, 5, 6))
    logits[0, 1, 1] = 3.0
    logits[0, 2, 2] = 2.0
    logits[0, 1, 4] = 1.0
    log_probs = torch.log_softmax(logits, dim=-1)
    input_ids = torch.tensor([[5, 5, 1, 2, 3]])
    answer_mask = torch.tensor([[False, False, True, True, True]])
    margins = canonical_null_margin_terms(
        log_probs,
        input_ids,
        answer_mask,
        torch.tensor([True]),
        Tokenizer(),
        [4],
    )
    expected = log_probs[0, 1, 1] - log_probs[0, 1, 4]
    assert margins.item() == pytest.approx(expected.item())
