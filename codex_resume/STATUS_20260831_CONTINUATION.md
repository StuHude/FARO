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
five-minute intervals.  The `.lock` is currently held and `submit.log` has
received a new heartbeat, so the retry runner is active in its own process
namespace.  Its PID cannot be validated with `kill -0` from this shell's
namespace; lock ownership and timestamped heartbeats are the authoritative
liveness evidence.  The shuffled PES branch remains locked until a valid
normal PES worker result exists.

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
- The retry runner is lock-protected and currently owns
  `logs/pes_submit/.lock`; do not launch a second copy.  The PID file is
  namespace-local, so liveness should be judged from lock ownership plus new
  `submit.log` heartbeats.

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

## Continuation check (2026-08-31 22:58 HKT)

- The full registered static suite was rerun in the worker-like conda
  environment with the pinned transformer source: `120 passed`.
- Focused PES/A-PES, submission, monitor, and action-margin contract tests
  passed `49 passed`; the candidate probe remains finite for all registered
  variants.
- The normal PES retry still has no `submitted` marker. A fresh
  `rjob list --namespace=ailab-dnacoding` attempt fails before API access with
  an unresolved configured proxy host, so no PES worker or holdout result is
  inferred.
- `submit_action_margin_diag.sh` now always reads the complete registered
  `rjob_tags.txt` allowlist, including GPU levels below eight; the obsolete
  alternate tag file was removed.
- The normal PES budget validator reports `actual_rows=5120`,
  `configured_rows=5120`, and `configured_steps=10`. SAMTok manifest and
  tail-GPPO preflight both return `status: ok` with the approved anchor.

These are implementation and launch-preflight checks only. The required
normal PES training, shuffled control, 512-row evaluations, and final
promotion remain pending until the dnacoding control plane is reachable.

## Evidence recheck (2026-08-31 23:20 HKT)

- The registered static suite passes `121 tests` after the submitter guards.
- A machine scan found 64 FARO bootstrap artifacts with exactly 512 paired
  rows and 20,000 repetitions; no non-official artifact violated this format.
- The working tree is clean. The PES retry remains single-instance and has no
  `submitted` marker; the control-plane DNS failure is unchanged.

## Current recheck (2026-08-31 23:30 HKT)

- The worker-like FARO contract subset was rerun with the `sa2va` environment:
  `130 passed in 9.46s`. This includes PES/A-PES scope, positive-tag,
  adaptive evaluation, training-budget, storage, and submitter guards.
- A full `pytest tests` collection in that environment remains unavailable
  because its installed `huggingface-hub==1.21.0` violates the pinned
  Transformers requirement `huggingface-hub>=0.34.0,<1.0` while importing two
  legacy PixVL tests. No dependency was installed or written into the
  repository; this is recorded as an environment gap rather than a FARO test
  pass or failure.
- `logs/pes_submit/.lock` is still held by the single retry process (its PID is
  outside this shell's PID namespace), and no second runner was started. The
  latest log heartbeat is `2026-08-31T23:24:13+08:00` with
  `control_plane_unavailable status=1`; `submitted` is absent.
- A direct `rjob list --namespace=ailab-dnacoding` and the requested proxy setup
  endpoint both remain unreachable due to DNS/proxy resolution. Consequently
  no PES job, checkpoint, worker metric, holdout, shuffled control, or final
  promotion is inferred.

## Current recheck (2026-08-31 23:40 HKT)

- The complete registered static contract selection was rerun after the
  retryer change: `131 passed in 9.07s`.
- The single PES retryer continues to hold `logs/pes_submit/.lock` and emitted
  `2026-08-31T23:39:17+08:00 control_plane_unavailable status=1`. No second
  runner was started and `submitted` remains absent.
- `submit_pes_after_ab_rejection.sh` now best-effort sources the requested
  internal proxy setup script after a failed control-plane query. A failed
  refresh is non-fatal and the original 300-second retry/backoff and all
  submission gates remain unchanged. This code applies on the next retryer
  restart; the currently locked process was not interrupted.
- The latest local commits are `9b73b46` (proxy refresh) and `3456aa8`
  (budget-audit documentation). These are local evidence only until GitHub
  connectivity is restored.

## Current recheck (2026-08-31 23:46 HKT)

- The same complete static selection, including the normal and shuffled PES
  transition scripts, passes `131 tests` after the proxy-refresh update.
- Both PES transition scripts now refresh the internal proxy configuration
  best-effort after a failed `rjob list`; shell syntax checks pass. The
  normal retry remains the only active instance, with its lock held and no
  `submitted` marker.
- The latest normal retry heartbeat remains
  `2026-08-31T23:39:17+08:00 control_plane_unavailable status=1`; DNS/API
  recovery has not occurred, so no training or evaluation artifact is claimed.

## Current recheck (2026-08-31 23:50 HKT)

- `tools/run_fepo_candidate_probe.py` completed successfully: all 17 registered
  candidate contracts report 5,120 rows, 10 optimizer steps, K=4, and finite
  local credit probes. `tools/run_apes_contract_probe.py` also passed with
  detached states `[0, 1, 2, 1]` and fixed shuffle seed `1907`.
- These probes are implementation-contract evidence only. They do not imply a
  model-quality result or unlock the shuffled branch.
- The normal retryer remains single-instance with its lock held. Its latest
  heartbeat is `2026-08-31T23:44:18+08:00 control_plane_unavailable status=1`,
  and no `submitted` marker exists; control-plane DNS remains unavailable.

## Resource recheck (2026-08-31 23:48 HKT)

- FARO storage is approximately `39G` total (`outputs` 34G, `evals` 4.4G,
  `logs` 43M), well below the 700G ceiling. No cleanup is required for the
  current experiment queue and no data was written under `PixVL_ailab`.
- The repository remains clean at 39 local commits ahead of `origin/main`.
- `rjob list --namespace=ailab-dnacoding` still fails before API access with
  the unresolved configured proxy; the PES retryer has no submission marker.
