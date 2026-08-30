# AB-FEPO preregistration: fixed action-budget native credit

## Hypothesis

AB-FEPO tests whether R18's native-relative geometry improvements become more
useful when unnecessary code edits carry a fixed cost. For each K=4 sibling
trajectory, the action count is the number of SAMTok code depths differing from
the native greedy trajectory. A joint cIoU and boundary-IoU improvement keeps
the R18 first-divergence rank credit; only edits beyond a fixed budget `B=2`
receive the deterministic multiplier `1/(1+0.10*excess)`. This is a training
credit regularizer, not an inference router, token truncation policy, teacher,
OPD target, counterfactual label, extra expert, or PixVL self-supervised loop.

## Fixed screen

- Continued-SFT SAMTok anchor, seed `17`, plain-rank arm.
- Exactly 5,120 training rows (2,560 no-target), 10 outer steps, K=4, two GPUs.
- Existing grammar-constrained per-prefix support calibration, FIFO geometry
  registry, native greedy reference, shared 32-row no-target sentinel, PPO
  ratio scope, and all validity/tail-risk gates remain unchanged.
- `action_budget=2` and `action_budget_excess_penalty=0.10` are fixed before
  any holdout access; no sweep is allowed.
- Logs must report action-change mean/p95, finite credit, support reach,
  positive-credit fraction, and full provenance. Action count is detached
  metadata only.

## Falsification and evaluation

AB-FEPO is eligible only after R35, BA-FEPO, and BS-FEPO have each been closed under
their preregistered decisions. Evaluate all 512 paired holdout rows with
20,000 paired bootstrap repetitions against R18 and forward-count-matched
continued-SFT. Require utility and positive cIoU non-inferiority, no-target
recall lower CI >= -0.01, zero invalid outputs, positive-mask rate >= 0.95,
and no registered slice regression. If action count decreases without a
utility/cIoU gain at matched null recall, close AB-FEPO as a decoding
regularizer. If any contract, support, or finite-credit gate fails, close
without tuning the budget or penalty.

BS-FEPO has now been closed by its complete negative holdout decision, so AB is
the next eligible isolated arm. Its lock-protected transition is allowed to
submit one training job once the dnacoding control plane is reachable. Any job
must use `dna-` naming, `ailab-dnacoding`, all positive tags from
`rjob_tags.txt`, and the existing 8 -> 6 -> 4 -> 2 -> 1 GPU evaluation
fallback. A queued or submitted job is not a quality result until the full
holdout and bootstrap gates complete.
