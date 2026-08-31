# FARO continuation status (2026-08-31)

This continuation was checked from the current `Faro_ailab/FARO` worktree.
No files were written under `PixVL_ailab`.

## Completed local evidence

- The conditional A-PES scope probe was run with the SAMTok worker Python:
  `tools/run_apes_contract_probe.py`.
- The probe passed with `manifest_rows=5120`, `steps=10`,
  `rollouts_per_prompt=4`, and fixed shuffle seed `1907`.
- Its detached probability-gap scope produced the registered synthetic states
  `[0, 1, 2, 1]` and finite shuffled scope.  This is a contract/probe result,
  not a model-quality result and does not promote A-PES.
- `tools/validate_training_budget.py` passed for the normal PES config using
  the approved read-only SAMTok checkpoint path:
  `actual_rows=5120 configured_rows=5120 configured_steps=10`.
- `tools/run_fepo_candidate_probe.py` completed with exit code `0`: all 17
  registered candidate contracts satisfy the minimum row/step/K checks, and
  all 13 available credit variants produced finite probe values.  This is
  implementation evidence only; it is not a training or quality result.
- The SAMTok worker environment reran the focused PES/A-PES and submission
  contract tests with `27 passed`.  The added PES regression checks verify
  that non-selected code depths receive exactly zero policy gradient,
  advantages remain detached, and seed-1907 shuffling changes only evidence
  state assignment.  These are local objective-contract checks, not model
  quality evidence.
- A CPU dummy-model rollout exposed and fixed a sampler indexing defect: the
  native-vs-sampled margin now gathers the sampled logit with its depth-local
  candidate index rather than the global vocabulary token id.  The regression
  covers effective-support sampling and action-term rescoring with offset
  code vocabularies; the focused PES suite passes `6 passed` after the fix.
- A consistency audit found the standalone A-PES probe had reversed the
  registered gap direction.  `tools/apes_probe.py` and its synthetic tests now
  use the same larger-gap-is-confident (`>=`) semantics as the trainer; the
  corrected probe still returns states `[0, 1, 2, 1]`.

## Control-plane recheck

The direct `rjob list --namespace=ailab-dnacoding` call still fails before API
access because the configured proxy host cannot be resolved.  Loading the
requested internal setup script was also unavailable because its host cannot
be resolved from this workspace.  No job was created, so no checkpoint,
worker metric, holdout, or bootstrap result is inferred.

The lock-protected normal PES retry state is stored at `logs/pes_submit/`; its
latest records continue to report `control_plane_unavailable status=1` at
five-minute intervals.  The shuffled PES branch remains locked until a valid
normal PES worker result exists.

## Next executable action

When dnacoding DNS/API access recovers, keep the existing normal PES retry
runner and submit the registered 2-GPU job with all positive tags from
`rjob_tags.txt`.  Apply the required evaluation fallback `8 -> 6 -> 4 -> 2
-> 1`, waiting five minutes at each non-terminal level.  Only a valid
5,120-row/10-step/K=4 worker artifact can unlock the fixed 512-row,
20,000-bootstrap normal-vs-anchor evaluation and then its shuffled control.
