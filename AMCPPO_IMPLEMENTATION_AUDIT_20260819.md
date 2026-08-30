# Standalone AM-CPPO Implementation Audit (2026-08-19)

## Scope and lineage

The standalone package under `Sa2VA/projects/samtok_selective` satisfies the
model-lineage boundary: it loads the original SAMTok base and the corrected
standalone `outputs/samtok_selective/continued_sft/adapter`. It does not import
PixVL trainers, checkpoints, routers, verifiers, or cycle/self-supervised
training. Its paired sampler keeps one target-present and one no-target row on
each rank. The registered smoke is two GPUs and 20 steps.

## Corrected defect

`canonical_null_margin_terms` previously subtracted the one-token
`<|mt_start|>` log probability from the *sum* of all tokens in
`"No target."`. That is not an action margin because its two sides have
different sequence lengths. It has been changed to

```text
log P(first token of "No target." at the answer boundary)
- log P(<|mt_start|> at the same boundary).
```

The complete canonical phrase is still validated. A regression test makes the
later phrase-token probabilities extreme and verifies that they cannot change
the first-action margin.

The already completed `fepo_ampcpo_smoke_2gpu` artifacts predate this fix.
Their mean reported margin (`12.82`) and almost-always-zero margin penalty do
not validate the corrected constraint. They must not be relabeled as a
correct AM-CPPO result.

## Remaining blocker: the positive objective is not on-policy PPO

The current `positive_actions_and_ciou` uses the GT mask answer as the input
sequence. It then chooses each code by argmax from teacher-forced logits. The
second predicted code is therefore conditioned on the GT first code, not on
the model's first predicted code. The decoded reward and optimized token log
probabilities do not describe one autoregressive trajectory.

In addition, `old_action_log_probs` is `action_log_probs.detach()` from the
same forward pass. The ratio is exactly one on every step and the observed clip
fraction is always zero. With one gradient update per batch, this is a
reward-weighted greedy imitation gradient, not a meaningful clipped PPO
update. The 20-step metrics confirm `clip_fraction=0` for all steps.

The existing 512-row result is consequently diagnostic only. Against the
equal-step SFT control it changes utility by `+0.00228` with 95% CI
`[-0.00356,+0.00848]`, positive cIoU by `+0.00065` with CI
`[-0.00911,+0.00927]`, and no-target recall by one row (`+0.00391`). It does
not pass the smoke promotion gate even before correcting the method label.

## Minimum patch for a valid RL smoke

Do not extend the current teacher-forced helper. Replace the positive path with
four explicit stages:

1. Build prompt-only multimodal inputs for each positive row.
2. Under `torch.no_grad()`, sample `K=4` complete autoregressive mask
   trajectories with a SAMTok grammar: first token `<|mt_start|>`, one token
   from codebook depth 0, one from depth 1, then `<|mt_end|>`. Every later token
   is conditioned on the sampled prefix. Record the sampling-policy log
   probability for the complete mask action.
3. Decode each complete trajectory, compute cIoU, and form leave-one-out or
   group-standardized advantages. Reject the batch for RL if all rewards are
   equal; keep the paired null CE/margin update.
4. Rebuild full prompt-plus-sampled-answer inputs and run one differentiable
   forward pass. Gather the same sampled action token log probabilities and
   optimize `-advantage * log_prob`. If multiple optimization epochs are used,
   retain the recorded sampling log probabilities and apply PPO clipping;
   with one epoch, remove the misleading clip term and call the objective
   constrained group-relative REINFORCE rather than CPPO.

The no-target half remains teacher-forced canonical null CE plus corrected
first-action margin. Do not generate negative rollouts because binary null
groups were already shown to have insufficient active-group rate.

## Concrete file changes for that patch

- `modeling.py`: add `build_prompt_inputs` and a grammar-constrained SAMTok
  mask sampler that returns token IDs and detached old log probabilities.
- `fepo_ampcpo_trainer.py`: replace `positive_actions_and_ciou`; expand each
  positive row to group size four, compute group advantages, then score sampled
  sequences in a differentiable forward. Preserve paired negative constraint
  rows in the same optimizer update.
- `configs/fepo_ampcpo_smoke_2gpu.py`: register `group_size=4`, sampling
  temperature/top-p, advantage normalization, and either `policy_epochs=1`
  with no PPO name or `policy_epochs>1` with a frozen old-log-prob ratio.
- `ampcpo_contract.py`: reject a CPPO label when group size is below two,
  sampling is disabled, old log probabilities are not frozen, or all policy
  epochs use the same current/detached tensor.
- `tests/test_samtok_standalone.py`: add deterministic fake-model tests that
  prove code 2 is conditioned on sampled code 1, group-constant rewards yield
  zero policy advantage, the ratio changes after a policy update, and null
  margin gradients remain present when RL advantage is zero.

## Safe execution gate

Before another rjob, run the CPU/static contract tests and a two-GPU one-step
grammar smoke. The full 20-step run may start only when logs show at least one
nonconstant rollout group, finite group advantages, valid four-token mask
trajectories, and (if PPO is retained) a nontrivial but bounded ratio. Then
evaluate all 512 rows against corrected continued-SFT and equal-step SFT. Do
not promote unless the frozen gate in `FEPO_NEXT_IDEAS_20260819.md` passes.
