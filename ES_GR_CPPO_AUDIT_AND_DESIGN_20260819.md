# Effective-Support GR-CPPO: Failure Audit and Minimal Successor

Date: 2026-08-19

Status: preregistered design only. This document does not authorize a trainer
change or an rjob submission.

## 1. Scope and evidence

This successor remains inside the fixed standalone FEPO boundary:

- one shared SAMTok mask-or-null policy;
- initialization from the frozen total-500 SFT adapter with model SHA256
  `7b409c9f2bc3cf2da61adb9c86270dcda6a1991082d5dca4bb1c8a593ea4dfed`;
- real autoregressive SAMTok grammar rollouts and plain cIoU reward;
- canonical no-target CE plus the existing first-null-token versus mask-start
  constraint;
- no PixVL weights or training method, no verifier, no router, no
  self-training loop, and no counterfactual rollout.

The registered plain GR-CPPO one-step artifact is
`outputs/samtok_selective/fepo_gr_cppo_one_step_2gpu/metrics.json`. Across its
two global positive groups it recorded:

| metric | observed |
|---|---:|
| rollouts per group | 4 |
| unique trajectories, mean | 1.0 |
| nonconstant reward groups | 0 / 2 |
| reward standard deviation | 0.0 |
| mean absolute advantage | 0.0 |
| epoch-2 ratio absolute deviation | 0.0009804 |
| grammar-valid fraction | 1.0 |

The failure precedes PPO. All four samples in each group select the same code
trajectory, so group standardization returns zero advantages and the positive
policy loss is exactly zero. The nonzero epoch-2 ratio movement does not
rescue the gate: it can only reflect the paired null update changing shared
parameters. It is not evidence of a reward-driven policy update.

The current implementation correctly samples later code tokens conditioned on
sampled prefixes, excludes forced boundary tokens from the constrained-policy
log probability, detaches rollout log probabilities, and reuses them across
two policy epochs. The next test should preserve those semantics. It should
change only the behavior distribution needed to expose more than one real
mask candidate.

## 2. Why a global temperature is not the next experiment

A fixed-temperature sweep would answer only which coefficient happened to
work for the sampled prompts. SAMTok confidence can differ substantially by
image, prompt, depth, and sampled prefix. One global temperature can leave a
sharp state deterministic while making an already uncertain state nearly
uniform. Repeatedly choosing it after observing the 512-row holdout would also
turn that holdout into tuning data.

Shannon effective support, `exp(H(q))`, is useful to report but is not the
right control target by itself. K-sample duplication is governed directly by
the collision probability `sum_a q(a)^2`. A large codebook can achieve a
moderate Shannon entropy by assigning small aggregate mass to many implausible
tail codes while retaining a dominant top action. That can raise
`exp(H(q))` without reliably producing useful distinct masks.

## 3. Candidate: ES-GR-CPPO

The minimal successor is **Effective-Support Grammar-Rollout Constrained PPO
(ES-GR-CPPO)**. It uses a per-sample, per-depth, per-sampled-prefix behavior
distribution. Its sole change from plain GR-CPPO is an automatically calibrated
exploration transform over the highest-ranked old-policy code actions.

Let `K=4` be the registered rollout-group size. At each code decision state
`s`, let `z_old(s)` be detached old-policy logits over the depth-specific valid
SAMTok code vocabulary.

1. Select `C_s`, the deterministic top `M=2K=8` actions under `z_old(s)`.
   Ties are broken by token id. The factor-two headroom ensures the target
   below is attainable at finite temperature; it is not selected from an
   evaluation result.
2. For temperature `tau >= 1`, define

   ```text
   q_old(a | s; tau) = softmax(z_old(s)[C_s] / tau)_a,  a in C_s,
                       0,                               otherwise.
   N2(q) = 1 / sum_a q(a | s)^2.
   ```

3. If `N2(q_old(tau=1)) >= K`, keep `tau_s=1`. Otherwise use deterministic
   float32 bisection to find the smallest `tau_s` for which `N2(q_old) >= K`,
   within tolerance `0.05`. Use the fixed numerical ceiling `tau_s <= 128`.
   Failure to reach the target at the ceiling fails the gate; it does not
   trigger a wider support or temperature sweep.
4. Independently sample each of the K trajectories from this distribution at
   each autoregressive code decision. A later support and temperature are
   recomputed from old-policy logits conditioned on that trajectory's sampled
   prefix. Mask-start and mask-end remain forced grammar actions.

`N2=K` is tied to the number of samples, not to downstream evaluation. For
K=4 it bounds the probability that two independent actions collide at a
controlled state to at most `1/4`. The top-8 support prevents a high
temperature from distributing probability across the full low-ranked
codebook. This is a behavior-policy intervention over real mask trajectories,
not a new label, synthetic negative, or counterfactual example.

The transform preserves the old logits' argmax. Evaluation therefore remains
ordinary deterministic SAMTok decoding without a temperature controller or
an inference-time branch.

## 4. Required log-probability and ratio semantics

For every sampled trajectory `y_i`, store at rollout time, for every code
depth `d`:

- sampled token `a_id`;
- old-prefix top-8 token-id set `C_id`;
- frozen scalar `tau_id`;
- detached `log q_old(a_id | s_id; C_id, tau_id)`.

The detached trajectory behavior log probability is

```text
log Q_old(y_i | x) = sum_d log softmax(
    z_old(s_id)[C_id] / tau_id
)[a_id].
```

During both policy epochs, rescore the sampled prefixes under current logits
using the *same stored support and temperature*:

```text
log Q_theta(y_i | x) = sum_d log softmax(
    z_theta(s_id)[C_id] / tau_id
)[a_id]
r_i(theta) = exp(log Q_theta(y_i | x) - log Q_old(y_i | x)).
```

The PPO clipped surrogate uses `r_i` and the frozen group-standardized cIoU
advantage. Supports and temperatures must not be recalibrated between policy
epochs. At rollout parameters the ratio is one, up to the already observed
mixed-precision/rescoring tolerance; after the first optimizer update it may
move.

This is PPO over the explicitly transformed grammar policy `Q`, whose logits
are the trainable SAMTok logits. It is **not** valid to sample from `Q_old` but
use `pi_theta / Q_old`, where `pi_theta` is the native temperature-one full
code policy: such ratios are already off one before any update and PPO
clipping no longer represents a trust region around the behavior policy. It
is also invalid to recompute current top-8 sets while retaining old behavior
log probabilities, because support changes create zero/infinite ratios.

Forced mask boundaries retain constrained probability one and remain outside
both log probabilities. Native temperature-one sampled-sequence log
probability and `KL(Q_old || pi_old)` should be logged only as diagnostics,
not substituted into the PPO ratio.

## 5. One-step hard gate

Use one fixed seed and one global optimizer step on exactly eight positive
training prompts (four per GPU), each with K=4 trajectories, paired with eight
no-target training rows. IDs are selected by a deterministic hash before
reward computation. This is still a one-step optimizer test, but its eight
groups prevent a two-prompt sampling accident from deciding the method.

The job passes only if all conditions below hold:

1. All 32 trajectories satisfy the two-code SAMTok grammar and decode to a
   finite mask reward in `[0, 1]`.
2. Every sampled code decision reaches `N2 >= 3.95`, or already has native
   top-8 `N2 >= 4`; no decision hits the `tau=128` ceiling below target.
3. At least six of eight groups contain at least two distinct complete code
   trajectories.
4. At least two of eight groups have reward standard deviation greater than
   `1e-6`, and global mean absolute advantage is finite and nonzero.
5. Relative to the native temperature-one greedy trajectory for the same
   prompt, at least one of the 32 sampled trajectories improves cIoU by more
   than `1e-4`. Diversity consisting only of worse alternatives is not
   evidence that geometry RL can improve the anchor.
6. Behavior log probabilities are finite and detached. Epoch-1 ratio mean is
   within `0.002` of one and every ratio is finite.
7. The positive policy loss has a nonzero finite gradient norm before adding
   null losses. This explicitly prevents a null-only update from passing the
   PPO gate.
8. After the first optimizer update, epoch-2 median absolute ratio deviation
   is greater than `1e-6`; clip fraction is at most `0.5`; canonical null CE
   and first-action margin loss are finite.

Required diagnostics are native and controlled `N2`, Shannon entropy,
temperature mean/median/p95/max, target-hit fraction, top-8 native probability
mass, `KL(Q_old || pi_old)`, unique trajectory count, reward range, fraction
improving over greedy, positive-only gradient norm, ratio quantiles, and clip
fraction. Aggregate every gate statistic across both ranks before deciding.

No 512-row evaluation follows a failed one-step gate.

## 6. Minimal controls and escalation

If the one-step gate passes, freeze the transform and run one 20-step arm.
Evaluate all 512 paired rows using ordinary temperature-one greedy decoding.
Compare against:

1. the frozen total-500 SFT anchor;
2. a continued-SFT control initialized from that anchor, matched for optimizer
   steps, training IDs, and expensive forward/backward compute;
3. the already completed plain temperature-one GR-CPPO one-step failure as a
   sampling-validity control. Do not run a meaningless 20-step plain arm whose
   positive advantages are known to be zero.

Only if ES-GR-CPPO survives the 20-step screen, add one non-swept adaptivity
control for a paper-scale experiment: fit a single global temperature on the
same frozen training-only probe IDs so that its *mean* `N2` matches the
adaptive arm, then freeze it. Adaptive versus mean-matched global temperature
tests whether state-specific effective support matters without tuning either
arm on the 512 holdout.

The 20-step arm must satisfy the shared FEPO screen in
`FEPO_NEXT_IDEAS_20260819.md`. In addition, at least 25% of rollout groups must
have nonconstant rewards, at least 10% must contain a sampled trajectory that
beats their native greedy reward, the support controller must have zero
failures, and the median epoch-2 ratio deviation must exceed `1e-6` with clip
fraction at most `0.5`. It advances to the frozen 100-step paper gate only if
it also beats the matched SFT control on selective utility point estimate and
does not lose positive cIoU.

## 7. Abandon conditions

Close ES-GR-CPPO, without changing `K`, `M`, the effective-support target,
temperature ceiling, seed, or reward, when any of these occurs:

- the one-step gate fails for reasons other than a demonstrated implementation
  bug;
- target support requires `tau > 128` or top-8 native mass is so small that
  sampled masks fail the finite decode/reward checks;
- diversity rises but no sampled candidate improves its native greedy cIoU;
- the positive-only policy gradient is zero or the correctly defined
  epoch-2 ratios do not move;
- the 20-step full-512 result fails the shared FEPO screen or fails to beat the
  matched SFT control;
- a surviving 100-step arm fails the preregistered paper gate.

A failure means the strong SFT anchor does not expose useful local mask-code
alternatives to group-relative RL under a bounded, rank-preserving exploration
policy. It should direct the project away from temperature engineering, not
toward a coefficient sweep. PASS-PO is also not the response to zero useful
positive candidates: it addresses unsafe realized updates only after a real
positive policy signal exists.

## 8. Claim boundary if successful

The defensible claim would be that selective pixel RL after strong SAMTok SFT
requires **state-adaptive effective candidate control** before group-relative
credit is usable. The method contribution is the collision-calibrated behavior
policy paired with a mask-or-null constraint, not a new tokenizer, a router,
or a general PPO invention. A result that does not beat matched SFT remains a
negative optimization finding and must not be packaged as a successful RL
framework.
