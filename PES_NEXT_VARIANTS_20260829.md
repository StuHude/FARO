# PES follow-up variants (registered queue, 2026-08-29)

This is a conditional queue, not authorization for concurrent jobs. The normal
PES-FEPO and its shuffled-evidence control must first complete worker gates,
the full 512-row image-disjoint holdout, and 20,000 paired bootstrap samples.
No threshold is tuned from that holdout. At most one follow-up is submitted.

## Common framework

Every arm is a single SAMTok policy initialized from the frozen continued-SFT
anchor. R18's native-relative joint cIoU/boundary-IoU rank remains the only
geometry credit; the selected evidence signal can only choose the token scope
that receives that credit. The no-target sentinel is a training-only hard
constraint. There is no PixVL trainer, teacher, OPD target, EMA, cycle,
counterfactual label, inference router, or second expert. Each arm requires
5,120 rows, at least 10 optimizer steps, K=4 grouped rollouts, finite ratios,
effective support, complete 512-row evaluation, zero invalid outputs, and
20,000 paired bootstrap repetitions.

## Variant M: mass-aware predicted evidence (only if PES is positive)

**Question.** Does the top-support probability add information beyond entropy
and native margin when deciding whether a local update is trustworthy? This is
the smallest follow-up inspired by V-Zero's detached uncertainty and
OpenWorldSAM's explicit abstention treatment.

**Fixed rule.** Keep PES entropy/margin states unchanged, then require mean
top-8 native support mass >= 0.70 for the confident one-depth scope and >=
0.25 for the ambiguous two-depth scope. Otherwise use the empty scope. Values
are fixed before submission; no sweep is allowed. The shuffled mapping is the
negative control.

**Promotion test.** The mass-aware arm must beat both entropy+margin PES and
matched continued-SFT on utility and positive cIoU without lowering the null
recall lower bound below -0.01. If it matches the shuffled control or loses
support coverage, close the uncertainty line.

## Variant A: probability-gap evidence (only if PES is rejected for weak action alignment)

**Question.** Does a probability gap measure whether the sampled action itself
is trustworthy more directly than a raw native-vs-sampled logit gap? This is a
strictly local correction motivated by V-Zero's trajectory discrimination, not
a new reward or an inference router.

**Fixed rule.** At each depth record detached native top-1 probability
`p_native` and sampled-action probability `p_sampled` under the calibrated
support distribution, then use `g = p_native - p_sampled`. Confident requires
mean entropy `< 0.35` and `g <= 0.10`; ambiguous requires non-confident and
(`entropy < 0.70` or `g <= 0.40`); otherwise scope is empty. The shuffled
state mapping with seed 1907 is mandatory.

**Promotion test.** Require action-state coverage, non-inferiority against
matched continued-SFT and the original PES, and a normal-versus-shuffled
difference on the complete 512-row/20k bootstrap. Thresholds are fixed before
training and cannot be selected from holdout results.

## Variant B: boundary-first scope (only if PES improves mean cIoU but not tails)

**Question.** Does assigning the verified joint credit to the first two changed
depths only on a detached boundary-hardness signal recover thin-object quality?
This isolates the boundary emphasis of Qwen3VL-Seg and DR2Seg without changing
the reward or data mixture.

**Fixed rule.** Use the same entropy+margin evidence state as PES. For state 0,
use depth one unless the detached boundary-IoU deficit of the native action is
in the registered upper quartile of the training sentinel, in which case use
the first two changed depths. States 1 and 2 retain PES scopes. The quartile is
computed on training rows only and frozen before the holdout.

**Promotion test.** Primary endpoint is the preregistered thin and
boundary-hard slice, with overall utility and null recall as non-inferiority
constraints. A mean-only gain closes this variant as a diagnostic regularizer.

## Variant N: null-safe evidence scope (only if PES loses no-target recall)

**Question.** Can the OpenWorldSAM/SenseNova-Vision capability-retention idea
be expressed as a constraint on scope rather than a new reward term?

**Fixed rule.** Keep PES scopes for positive trajectories. If a training group
contains a sentinel margin in the fixed lower decile below the anchor budget,
set all positive geometry scopes in that group to empty for the next optimizer
update; null CE and margin losses remain unchanged. The rule is a fixed
training-time guard, not an inference abstention threshold.

**Promotion test.** This arm is eligible only after a reproducible PES null
recall regression. It must restore the registered null lower bound while
retaining positive cIoU non-inferiority; otherwise close it without tuning.

## Decision order

1. PES normal and shuffled control.
2. If normal is rejected, close the branch unless one of M/B/N's explicit
   trigger conditions is met; then run exactly one triggered arm.
3. If normal passes, run the shuffled-control comparison and promote only when
   the evidence mapping, not merely scoped credit, is causal.
4. Only a promoted survivor receives 100-step confirmation and official
   RefCOCO/GRefCOCO transfer evaluation.

This queue keeps the ten-paper synthesis as one falsifiable credit-scope
framework instead of a collection of independently tuned routers.
