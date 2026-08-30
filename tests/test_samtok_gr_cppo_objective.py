from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from projects.samtok_selective.fepo_gr_cppo_trainer import (  # noqa: E402
    build_rollout_prompt_batch,
    canonical_null_ce_and_first_action_margin,
    clipped_policy_loss,
    group_standardized_advantages,
    sample_grammar_rollouts,
    score_sampled_sequences,
)


GRAMMAR_IDS = (1, [[3, 4], [5, 6]], 2)


class FakeProcessor:
    def __init__(self):
        self.calls = []

    @staticmethod
    def apply_chat_template(messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"][1]["text"]

    def __call__(self, *, text, images, padding, return_tensors):
        self.calls.append(
            {
                "text": text,
                "images": images,
                "padding": padding,
                "return_tensors": return_tensors,
            }
        )
        return {
            "input_ids": torch.tensor([[10, 11]] * len(text)),
            "attention_mask": torch.ones(len(text), 2, dtype=torch.long),
            "pixel_values": torch.arange(len(images), dtype=torch.float32).unsqueeze(1),
            "image_grid_thw": torch.ones(len(images), 3, dtype=torch.long),
        }


class PrefixRecordingModel(torch.nn.Module):
    """Makes each registered grammar action effectively deterministic."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(7))
        self.seen_input_ids = []

    def forward(self, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        self.seen_input_ids.append(input_ids.detach().clone())
        batch, length = input_ids.shape
        logits = self.bias.view(1, 1, -1).expand(batch, length, -1).clone()
        last = input_ids[:, -1]
        # Before an answer, prefer mask start; at the two code depths, prefer
        # the first valid code; after depth two, prefer mask end.
        preferred = torch.where(
            last.eq(11),
            torch.tensor(1, device=input_ids.device),
            torch.where(
                last.eq(1),
                torch.tensor(3, device=input_ids.device),
                torch.where(
                    last.eq(3),
                    torch.tensor(5, device=input_ids.device),
                    torch.tensor(2, device=input_ids.device),
                ),
            ),
        )
        rows = torch.arange(batch, device=input_ids.device)
        logits[rows, -1, preferred] += 80.0
        return SimpleNamespace(logits=logits)


def test_group_standardized_advantages_and_zero_variance_behavior():
    advantages = group_standardized_advantages(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert advantages.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert advantages.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-6)
    assert torch.equal(
        group_standardized_advantages(torch.full((4,), 3.0)), torch.zeros(4)
    )


def test_processor_reencodes_one_text_image_copy_per_rollout():
    processor = FakeProcessor()
    image = object()
    batch = build_rollout_prompt_batch(
        processor,
        {"prompt_text": "segment target", "image": image},
        4,
        device=None,
    )
    call = processor.calls[-1]
    assert call["text"] == ["segment target"] * 4
    assert len(call["images"]) == 4
    assert all(item is image for item in call["images"])
    assert call["padding"] is True
    assert call["return_tensors"] == "pt"
    assert batch["input_ids"].shape[0] == 4
    assert batch["pixel_values"].shape[0] == 4


def test_rollouts_are_autoregressive_and_behavior_logprob_is_detached():
    model = PrefixRecordingModel()
    prompt_inputs = {
        "input_ids": torch.tensor([[10, 11]] * 4),
        "attention_mask": torch.ones(4, 2, dtype=torch.long),
    }
    sequence_token_ids, sampled_codes, behavior_log_probs = sample_grammar_rollouts(
        model, prompt_inputs, GRAMMAR_IDS, temperature=1.0
    )

    expected = torch.tensor([[1, 3, 5, 2]] * 4)
    assert torch.equal(sequence_token_ids.cpu(), expected)
    assert sampled_codes == [[0, 2]] * 4
    assert behavior_log_probs.shape == (4,)
    assert not behavior_log_probs.requires_grad
    # The depth-one-code forward must be conditioned on start + sampled
    # depth-zero code, rather than teacher-forced from a fixed answer.
    assert any(
        seen.shape[1] >= 4 and torch.equal(seen[0, -2:].cpu(), torch.tensor([1, 3]))
        for seen in model.seen_input_ids
    )


def test_differentiable_rescoring_conditions_on_complete_sampled_prefix():
    model = PrefixRecordingModel()
    prompt_inputs = {
        "input_ids": torch.tensor([[10, 11]] * 4),
        "attention_mask": torch.ones(4, 2, dtype=torch.long),
    }
    sampled = torch.tensor([[1, 3, 5, 2]] * 4)
    scores = score_sampled_sequences(
        model, prompt_inputs, sampled, GRAMMAR_IDS, temperature=1.0
    )
    assert scores.shape == (4,)
    assert scores.requires_grad
    assert torch.equal(model.seen_input_ids[-1][:, -4:].cpu(), sampled)
    (-scores.mean()).backward()
    assert model.bias.grad is not None


def test_forced_mask_boundaries_are_excluded_from_policy_logprob():
    model = PrefixRecordingModel()
    prompt_inputs = {
        "input_ids": torch.tensor([[10, 11]] * 4),
        "attention_mask": torch.ones(4, 2, dtype=torch.long),
    }
    sampled = torch.tensor([[1, 3, 5, 2]] * 4)
    first = score_sampled_sequences(
        model, prompt_inputs, sampled, GRAMMAR_IDS, temperature=1.0
    )
    with torch.no_grad():
        model.bias[1] = -30.0
        model.bias[2] = 30.0
    second = score_sampled_sequences(
        model, prompt_inputs, sampled, GRAMMAR_IDS, temperature=1.0
    )
    assert torch.allclose(first, second)


def test_post_update_ratio_can_move_and_clip_on_second_policy_epoch():
    behavior = torch.zeros(2)
    current = torch.tensor([0.0, torch.log(torch.tensor(1.5))], requires_grad=True)
    advantages = torch.ones(2)
    loss, ratios, clip_fraction = clipped_policy_loss(
        current, behavior, advantages, clip_epsilon=0.2
    )
    assert torch.allclose(ratios, torch.tensor([1.0, 1.5]))
    assert loss.item() == pytest.approx(-1.1)
    assert clip_fraction.item() == pytest.approx(0.5)


def test_null_ce_covers_phrase_while_margin_compares_only_first_actions():
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

    input_ids = torch.tensor([[9, 9, 1, 2, 3]])
    attention = torch.ones_like(input_ids)
    answer_mask = torch.tensor([[False, False, True, True, True]])
    no_target = torch.tensor([True])
    logits = torch.zeros(1, 5, 7)
    logits[0, 1, 1] = 3.0
    logits[0, 1, 4] = 1.0
    logits[0, 2, 2] = 2.0
    logits[0, 3, 3] = 2.0

    null_ce, margins = canonical_null_ce_and_first_action_margin(
        logits,
        input_ids,
        attention,
        answer_mask,
        no_target,
        Tokenizer(),
        mask_start_id=4,
    )
    boundary_log_probs = torch.log_softmax(logits[0, 1], dim=-1)
    assert margins.item() == pytest.approx(
        (boundary_log_probs[1] - boundary_log_probs[4]).item()
    )

    degraded = logits.clone()
    degraded[0, 2, 2] = -8.0
    degraded_ce, degraded_margins = canonical_null_ce_and_first_action_margin(
        degraded,
        input_ids,
        attention,
        answer_mask,
        no_target,
        Tokenizer(),
        mask_start_id=4,
    )
    assert degraded_ce.item() > null_ce.item()
    assert degraded_margins.item() == pytest.approx(margins.item())
