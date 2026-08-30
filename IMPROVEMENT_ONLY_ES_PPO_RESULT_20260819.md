# Improvement-Only ES-PPO Result (2026-08-19)

## Registered hypothesis

Candidate E tested whether positive-only credit transfers sampled mask-code
improvements to the deterministic SAMTok policy more effectively than
group-standardized advantages:

```text
advantage = relu(sampled_cIoU - native_greedy_cIoU - 1e-4)
advantage /= mean(positive advantages in the group)
```

It retained the fixed-null effective-support exploration, K=4, two PPO
epochs, canonical no-target CE, first-action margin term, optimizer, data, and
continued-SFT initialization. It used no router, PixVL training component,
cycle training, counterfactual rollout, boundary reward, FIFO, or hard-slice
label.

## Training validity

The two-GPU 20-step run completed successfully at:

```text
outputs/samtok_selective/fepo_improvement_only_es_ppo_20step_2gpu
```

Across 20 outer steps it produced 160/160 nonconstant-reward groups and
160/160 multitrajectory groups. There were 132 sampled trajectories above the
native greedy cIoU. The mean active-advantage fraction was `0.20625` and its
per-step minimum was `0.0625`. The minimum positive-policy gradient norm was
`3.5720`, effective-support target hit rate was `1.0`, maximum PPO clip
fraction was `0.125`, and the maximum epoch-two median ratio deviation was
`0.05142`. The optimization mechanism therefore passed its validity screen.

## Full 512-row evaluation

The complete image-disjoint holdout contains 256 positive and 256 no-target
rows. The deterministic evaluation is:

```text
evals/standalone_improvement_only_es_ppo_20step_512.json
```

| metric | Candidate E |
|---|---:|
| selective utility | 0.788334 |
| positive cIoU | 0.771980 |
| no-target explicit recall | 0.804688 |
| positive mask rate | 1.000000 |
| invalid output rate | 0.000000 |

Paired 20,000-repeat bootstrap comparisons are:

| reference | utility delta (95% CI) | positive cIoU delta (95% CI) | null-recall delta (95% CI) |
|---|---:|---:|---:|
| continued-SFT anchor | +0.004590 [-0.000249, +0.011174] | +0.001368 [-0.001397, +0.005843] | +0.007812 [0, +0.019531] |
| matched-data/update SFT | +0.007772 [+0.001791, +0.015686] | -0.000081 [-0.000528, +0.000223] | +0.015625 [+0.003906, +0.031250] |
| fixed-null ES-GR-CPPO | -0.005294 [-0.012937, +0.000607] | +0.001130 [-0.001874, +0.004835] | -0.011719 [-0.027344, 0] |

Relative to fixed-null ES, six positive raw outputs changed: two improved,
two degraded, and two changed only their serialization. Five negative raw
outputs changed; three were one-way regressions from the canonical null output
to a mask. Candidate E's exact canonical positive response rate was
`253/256 = 0.988281`, although all positive rows remained valid masks under
the permissive evaluator.

## Decision

Candidate E is closed and must not receive 100 steps. It improves utility over
the matched SFT control, demonstrating that the RL update is not equivalent to
continued supervised fitting, but it is worse than fixed-null ES and violates
the registered null-risk noninferiority gate: the null-recall confidence
interval lower bound is `-0.027344`, below the allowed `-0.01`.

The failure is informative. Positive-only advantages transfer geometry on a
small number of rows, but removing negative within-group credit also weakens
selective calibration. The next hypothesis must preserve the stronger
fixed-null behavior while testing a more direct greedy-policy transfer
mechanism. It must not tune the improvement threshold, K, support target,
reward weights, or seed on this holdout.
