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
- The standalone budget validator now inserts the repository's `Sa2VA` path
  itself, so direct root-level audits no longer depend on an inherited worker
  `PYTHONPATH`.  With the explicitly approved SAMTok checkpoint it reports
  `actual_rows=5120 configured_rows=5120 configured_steps=10`; it still fails
  closed when no checkpoint environment is supplied.
- The complete `tests/test_*static.py` suite was rerun in the worker-like
  `sa2va` environment with the pinned transformer source: `120 passed`.  This
  covers all registered candidate contracts, SAMTok-only/path guards, positive
  tags, training budgets, adaptive GPU fallback, and PES transition gates.
- The actual worker preflight command was run locally with the approved
  SAMTok anchor, 5,120-row manifest, and PES config: both
  `projects.samtok_selective.manifests guard` and
  `tail_gppo_contract --skip-model-hash` returned `status: ok`.  This confirms
  the launch-time path/contract checks, but is not GPU training evidence.
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
five-minute intervals.  The previous retry process is not currently alive in
this isolated workspace (the stale `pid` is not a running submitter), so no
claim of continuous background polling is made.  The shuffled PES branch
remains locked until a valid normal PES worker result exists.

## Next executable action

When dnacoding DNS/API access recovers, keep the existing normal PES retry
runner and submit the registered 2-GPU job with all positive tags from
`rjob_tags.txt`.  Apply the required evaluation fallback `8 -> 6 -> 4 -> 2
-> 1`, waiting five minutes at each non-terminal level.  Only a valid
5,120-row/10-step/K=4 worker artifact can unlock the fixed 512-row,
20,000-bootstrap normal-vs-anchor evaluation and then its shuffled control.

## Continuation check (2026-08-31 22:20 HKT)

- Rechecked the repository after the documentation refresh; `main` is clean
  and contains the new `118a876` README commit in addition to the 13 pending
  commits relative to `origin/main`.
- Re-ran the worker-like static suite: `120 passed`.  The candidate contract
  probe reported 17 registered contracts and 13 finite credit variants; the
  A-PES probe reported the fixed 5,120-row/10-step/K=4 contract and seed 1907.
  These remain offline implementation checks only.
- The latest `rjob list --namespace=ailab-dnacoding` retry still fails before
  API access with `ProxyError`/unresolved
  `httpproxy-headless.kubebrain.svc.pjlab.local`; `logs/pes_submit/submitted`
  is absent.  No PES job, checkpoint, holdout, or bootstrap artifact is
  inferred from this failure.
- A detached retry process cannot persist across this workspace's isolated
  tool sessions (`tmux` is unavailable and detached children are reaped), so
  the retry script must be relaunched in a live session after control-plane
  access is restored.

## Launch preflight recheck (2026-08-31 22:24 HKT)

- The PES manifest remains present with exactly 5,120 nonempty JSONL rows.
- `validate_training_budget.py` returned
  `training_budget_ok actual_rows=5120 configured_rows=5120 configured_steps=10`.
- With the approved read-only SAMTok anchor, both
  `projects.samtok_selective.manifests guard` and
  `tail_gppo_contract --skip-model-hash` returned `status: ok`.
- DNS remains unavailable for the configured proxy, so this preflight does
  not create an rjob or imply a checkpoint/quality result.
- The repository-local `outputs` and `logs` files occupy approximately 33.68
  GiB, so the 700G storage ceiling is not currently the limiting condition.
