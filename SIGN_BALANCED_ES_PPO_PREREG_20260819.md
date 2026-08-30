# Sign-Balanced Greedy-Relative ES-PPO Preregistration (2026-08-19)

## Candidate I hypothesis

Candidate E showed that positive-only greedy-relative credit can move geometry
but loses selective stability. Candidate H restored exactly the fixed-null ES
decisions with signed credit, yet negative trajectories occurred roughly three
times as often as positive trajectories and all changed positive outputs
degraded. Candidate I tests whether this is a sign-mass imbalance.

Within each K=4 group, let `p=relu(delta)` and `n=relu(-delta)`, where
`delta=sampled_cIoU-native_greedy_cIoU`. If both signs exist:

```text
advantage = (K/2) * (p / sum(p) - n / sum(n))
```

Thus positive and negative advantages each receive half the group's L1 mass,
while mean absolute advantage remains exactly one. A single-sign group is
normalized to mean absolute value one; an all-zero group remains zero. This
adds no fitted coefficient.

All data, SAMTok initialization, K=4 effective-support sampler, support target,
plain cIoU reward, PPO clipping, two epochs, null CE/margin, optimizer, and seed
remain fixed. There is no router, PixVL training component, preference pair,
counterfactual example, boundary mix, FIFO, hard label, or inference module.

Run a two-GPU one-step validity test, then at most one frozen 20-step run and a
complete 512-row evaluation. Require both advantage signs, unit mean absolute
advantage, nonzero gradients, epoch-two ratio change, support hit `1.0`, and
clip at most `0.5` in one-step. Promotion uses the same fixed-null ES and
matched-SFT noninferiority, null-risk, mask-rate, invalid-rate, and canonical
response gates. No sign weight, normalization, K, seed, support, or LR sweep is
allowed.

## Frozen result

The one-step mechanism gate passed, so the single registered 20-step run was
evaluated on all 512 rows (256 positive and 256 negative):

| policy | selective utility | positive cIoU | no-target recall |
|---|---:|---:|---:|
| matched SFT | 0.780562 | 0.772061 | 0.789063 |
| fixed-null ES | 0.793628 | 0.770850 | 0.816406 |
| sign-balanced ES | 0.793129 | 0.769852 | 0.816406 |

Against fixed-null ES, selective utility changed by `-0.000499` with paired
95% CI `[-0.001406, 0]`; positive cIoU changed by `-0.000998` with CI
`[-0.002811, 0]`; no-target recall was identical.  Against matched SFT, the
utility gain was `+0.012567`, but positive cIoU was `-0.002209`.  The registered
promotion gate therefore failed.  This branch is closed and no sign-ratio
sweep is allowed.

## Raw-output and canonical-format audit

The final paired audit compares decoded text, not only aggregate evaluator
fields.  A positive response is canonical only when it contains exactly one
depth-2 SAMTok mask sequence, optionally followed by `<|im_end|>`; a negative
response is canonical only when it contains `No target.`, again with only an
optional `<|im_end|>` suffix.

| policy | canonical all | canonical positive | canonical negative |
|---|---:|---:|---:|
| frozen SFT anchor | 458 / 512 | 254 / 256 | 204 / 256 |
| fixed-null ES | 459 / 512 | 250 / 256 | 209 / 256 |
| sign-balanced ES | 459 / 512 | 250 / 256 | 209 / 256 |

Relative to fixed-null ES, sign balancing changed exactly four raw outputs:
three positive masks and one negative mask.  All three positive changes reduced
cIoU (`-0.031956`, `-0.015671`, and `-0.207938`), while the changed negative
remained a non-null mask and therefore remained incorrect.  Canonical-format
counts were identical between the two methods.  Relative to the frozen SFT
anchor, sign-balanced ES changed 17 outputs (eight positive and nine negative),
but four positive responses also drifted into permissively accepted JSON/code
fence wrappers.  Thus neither permissive parsing nor aggregate ties conceal a
sign-balanced improvement.
