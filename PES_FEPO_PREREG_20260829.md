# PES-FEPO preregistration: predicted-evidence scope allocation

## Motivation

R18 is the only robust positive reference. BA, BS, and AB each changed a
single downstream ingredient of the same native-relative reward and all lost
against matched continued-SFT. R15 also showed that a fixed earliest-depth
decay is not a causal explanation. The next test must therefore change the
credit *scope*, using information available to the policy, rather than add a
new reward transform, data mixture, or action penalty.

## Hypothesis

For each K=4 rollout group, detached per-depth evidence (native-vs-sampled
logit margin and calibrated entropy) predicts whether an update should be
localized or broadened. A confident trajectory should update only its first
changed SAMTok code depth; an ambiguous trajectory should update its first two
changed depths; an unsupported trajectory receives no positive geometry credit.
The sign and magnitude of credit remain exactly R18's native-relative joint
cIoU/boundary-IoU improvement. This is a single mask-or-null policy, not an
inference router, teacher, OPD target, counterfactual label, or PixVL cycle.

## Fixed implementation

- SAMTok continued-SFT anchor, seed 17, plain-rank arm, K=4.
- Exactly 5,120 training rows (2,560 no-target), 10 optimizer steps.
- Per-depth evidence is detached and computed from the existing calibrated
  support logits; thresholds are fixed before training: confident if mean
  normalized entropy < 0.35 and native margin >= 1.0, ambiguous otherwise if
  entropy < 0.70 or margin >= 0.25, unsupported otherwise.
- Confident scope is one changed depth; ambiguous scope is two changed depths;
  unsupported scope is empty. Scope masks are applied to token-level PPO log
  probabilities, while the rollout behavior probabilities use the same frozen
  masks. No scalar credit rescaling is added.
- A deterministic shuffled-evidence mapping is a separate negative control;
  it preserves all rewards and masks but permutes evidence states within the
  group using seed 1907.

## Falsification

Require worker validity, finite ratios, at least 20% support in every nonempty
state, exactly 512 image-disjoint holdout rows (256 positive/256 no-target),
zero invalid outputs, and 20,000 paired bootstrap repetitions. Compare against
R18-100 and forward-count-matched continued-SFT. Promotion requires utility
and positive cIoU non-inferiority, no-target recall CI lower bound >= -0.01,
and no thin/boundary-hard slice regression. If the shuffled control matches
PES, or if evidence-state coverage is too sparse, close the branch without
threshold tuning.

## Resource contract

The eventual training job must use the existing SAMTok TB-GPPO wrapper with
5,120 rows and 10 steps, `dna-` naming, namespace `ailab-dnacoding`, and every
positive tag in `rjob_tags.txt`. Its evaluation uses the fixed 8 -> 6 -> 4 ->
2 -> 1 GPU ladder with 300-second waits. All outputs remain under `Faro_ailab`
and below the 700G limit.

PES-FEPO is a design hypothesis until its token-level scope implementation and
offline contract probe pass. No job is authorized concurrently with another
candidate.

## Implementation update (2026-08-29)

The trainer now preserves detached per-depth behavior log-probabilities and
uses `clipped_scope_policy_loss` when PES mode is selected. Evidence states
and scope lengths are emitted in each rollout summary, while legacy arms keep
the scalar PPO path. The config and submit wrapper enforce the 5,120-row,
10-step contract and the positive-tagged `dna-` rjob. A lock-protected
transition retries every 300 seconds after the completed AB rejection, so the
PES job is submitted once the control plane recovers without duplication.
The final training validity gate additionally requires at least two observed
evidence states, with every observed state covering at least 20% of rollout
groups; this is checked before any holdout adapter is accepted.

Before submission, the rollout evidence implementation was audited and aligned
with this registration: its detached margin is now
`max(native_candidate_logits) - sampled_code_logit`, rather than a native
top-1/top-2 margin. The previously unused mass-threshold fields were removed,
so state assignment is explicitly mean normalized entropy plus native-vs-
sampled margin. Coverage counters accumulate local counts before the final
distributed reduction, making reported absolute counts globally correct.

The shuffled-evidence transition is validity-gated as well: it may be
submitted only when normal PES is `finished` and its worker validity,
effective-support, tail-risk, and PES-coverage gates all pass. A failed normal
worker run closes the branch without spending compute on an uninterpretable
control.
