# Constrained Policy-Improvement Audit

## Decision

Do not submit a GPU run yet. V6 failed the preregistered utility gate, while
continued-SFT already improves official GRefCOCO cIoU from 68.80 to 70.85.
Therefore any RL arm must show improvement over continued-SFT, not only over
the raw SAMTok base.

The previous `samtok_selective_risk_constrained_rl_2gpu.py` was not a real
constraint method: it only changed negative CE/KL coefficients. The trainer
had no outcome-specific budget, violation, dual state, or constraint logging.

## Implemented Smoke Boundary

`risk_constraints.py` now defines a differentiable constraint over explicit
negative refseg rows (`meta.no_target=true`):

```text
v_null = relu(mean(CE_negative) - budget - epsilon)
lambda <- project(lambda + dual_lr * v_null, [0, lambda_max])
L <- L_existing + lambda * v_null
```

The trainer adds this term once per optimizer batch and logs `null_loss`,
`null_violation`, `null_lambda`, `null_budget`, `null_constraint_active`, and
`null_kl_mean`. This is a dual-Lagrangian smoke only; `projection=false` means
it is not a gradient-projection or exact trust-region guarantee.

## Required Calibration Before GPU

The budget must be measured from a frozen original-SAMTok base rollout on the
same image-disjoint positive/negative schema. Existing runs cannot provide it:
their metrics omit null CE and per-outcome KL. The configured `1.5` budget and
`0.1` epsilon are placeholders for a smoke configuration, not evidence.

## Valid Experiment Gate

Use fixed data, seed, decoder, steps, group size, and original SAMTok
initialization. Compare raw base, continued-SFT, existing null-aware control,
and constrained RL. Promote only if constrained RL improves positive cIoU by at
least `+0.015` over the matched control with paired 95% CI excluding zero,
keeps no-target recall within `-1 pp` of control, has null violation rate below
5%, and stays within the preregistered KL budget. A no-target-only gain is a
failure. No GPU submission is authorized by this audit until the base null-loss
calibration is available.
