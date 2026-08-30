# R20 preregistration: asymmetric signed native-relative credit

## Status

R20 is a pending, single-point follow-up to R18. It must not be submitted
until the complete official RefCOCO and GRefCOCO transfer scores identify a
concrete regression or headroom related to harmful sampled masks. If R18 is
adequate on both transfers, R20 is closed without a run.

## Hypothesis

R18 gives positive credit to a sampled mask only when both cIoU and boundary
IoU beat the frozen native-greedy mask, but leaves regressions and mixed-axis
trade-offs neutral. R20 tests whether a signed native-relative signal can
suppress harmful local code choices while retaining useful trade-offs.

This is motivated by DR2Seg's stable ranked/continuous geometry supervision
and the explicit positive/negative separation used by Qwen3VL-Seg and
OpenWorldSAM. It does not claim any of their architectures or data.

## Fixed method change

- Compute `0.75 * (cIoU-native_cIoU) + 0.25 *
  (boundaryIoU-native_boundaryIoU)` with fixed `beta=0.25`.
- Preserve the sign for positive, mixed, and regressive trajectories; only
  unchanged native codes and values inside the `1e-4` deadband are neutral.
- Assign the signed signal to the first changed code depth with factor
  `0.85**depth`.
- Normalize signed active credit by its detached mean absolute value.
- Keep K=4 effective-support rollouts, the unified 32-row no-target sentinel,
  tail-risk budgets, and the frozen continued-SFT SAMTok anchor.

There are no PixVL weights, OPD targets, self-supervised cycles,
counterfactual labels, inference routers, or extra experts.

## Required run and gate

The only permitted screen uses `egfepo_train_5120.jsonl` (5,120 rows), 10
outer steps, 20 policy epochs, and two GPUs. It evaluates all 512 paired
holdout rows with 20,000 paired bootstrap resamples. The run must retain
zero invalid outputs, positive-mask rate at least `0.95`, non-inferior
no-target recall, and all support/tail/sentinel validity gates.

Promotion requires positive cIoU delta at least `+0.015`, utility delta at
least `+0.02` with a strictly positive bootstrap lower bound, and
non-inferiority to R18 on positive cIoU and utility. No beta or temperature
sweep is allowed. Any failed validity or promotion gate closes R20.

## Run record (2026-08-28)

- Job: `dna-fepo-signed-native-depth-local-beta025-1-0e587` (dnacoding,
  two GPUs; one duplicate submission was stopped immediately).
- Training completed 10 outer steps / 20 policy epochs from the frozen
  `continued_sft_to500` SAMTok adapter on all 5,120 rows.
- `validity_gate.passed=true`; effective-support, sentinel, tail-risk,
  representation, nonconstant-reward, and epoch-2 ratio gates all passed.
- Final training rollout means: cIoU `0.69071`, boundary IoU `0.27395`,
  reward `0.54492`; signed-credit active fraction `0.125`.
- The complete 512-row holdout was evaluated with 20,000 paired bootstrap
  resamples. Relative to R18, positive cIoU improved by `+0.001340`
  (95% CI `[0,+0.004020]`), selective utility by `+0.004576` (CI
  `[0,+0.011106]`), and no-target recall by `+0.007813` (CI
  `[0,+0.019531]`). Invalid output remained `0.0` and positive-mask rate
  `1.0`.
- The preregistered promotion thresholds were not met. R20 is closed without
  any beta, temperature, or learning-rate sweep. R18 remains the selected
  method; R20 is retained as a signed-negative ablation.
