# R34 preregistration: soft native-dominance depth-local FEPO

R21/R28 show that the native midrank signal is quantized for K=4 rollouts:
small geometry differences often map to the same rank. R34 tests whether a
fixed-temperature continuous dominance score gives the shared SAMTok policy a
more useful geometry gradient without changing the policy, decoder, or null
risk constraint.

For each sampled mask, let `g_c` and `g_b` be cIoU and boundary-IoU gains over
the native greedy mask. A rollout is eligible only when both gains exceed
`1e-4`. Its detached credit is

```text
0.5 * [tanh(g_c / 0.02) + tanh(g_b / 0.02)] * 0.85^(first_changed_depth)
```

Credits are normalized by the active-group mean. No negative, mixed-axis, or
unchanged trajectory receives credit. The fixed temperature is registered
before training and is not selected on the holdout.

## Fixed screen

- Frozen SAMTok continued-SFT-to-500 adapter, seed 17.
- Exactly 5,120 training rows, 10 outer steps, K=4 rollouts, two policy
  epochs, two GPUs.
- Existing effective-support grammar, shared 32-row sentinel, null CE and
  first-action margin gates.
- Complete 512-row paired holdout and 20,000 paired bootstrap repetitions.

The matched reference is R18 native-relative rarity-free depth-local FEPO.
Promotion requires the strict CI-corrected utility gate, positive cIoU
non-inferiority, no-target non-inferiority, canonical-output and invalid-rate
checks, plus fixed geometry slices. If the screen fails, R34 is closed and no
temperature sweep or longer run is authorized.

The mechanism is a single SAMTok mask-or-null policy. PixVL weights, OPD,
self-supervised cycles, routers, experts, counterfactual views, and inference
routing are excluded.
