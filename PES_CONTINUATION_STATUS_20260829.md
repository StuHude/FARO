# PES-FEPO continuation status (2026-08-29)

The registered PES normal arm remains the only pending training submission.
Its contract is SAMTok-only, 5,120 rows, 10 optimizer steps, K=4, and the
native-relative joint cIoU/boundary-IoU reward with detached entropy/action
evidence scope. The shuffled evidence arm remains locked until normal PES
passes worker validity, effective-support, tail-risk, and PES-coverage gates.

The dnacoding control plane still fails DNS resolution for `h.pjlab.org.cn`.
The lock-protected transition therefore records a retry every 300 seconds and
has not created an rjob or consumed GPU resources. All submitters use the
namespace `ailab-dnacoding`, `dna-` names, and every positive tag in
`rjob_tags.txt`. Evaluation, once a checkpoint exists, retains the fixed
8 -> 6 -> 4 -> 2 -> 1 GPU ladder with 300-second waits.

This continuation fixed stale transition markers: `runner_started` is now
validated against a live PID and command line, while transition scripts write
and clean their PID files. A killed shell can therefore resume after the
control plane recovers without duplicate submission; the transition flock is
still the final idempotency guard.

Offline verification passed for the complete PES config/probe: 5,120 rows,
10 steps, K=4, all three evidence states, detached scope masks, and finite
scoped PPO gradients. No holdout or GPU result exists yet, so no new variant
has been promoted or tuned.

As a conditional follow-up only, an isolated CPU A-PES probe checks the
probability-gap formulation `p_native - p_sampled` suggested by the literature
audit. It reports all three states and deterministic seed-1907 shuffling, but
does not alter the registered PES trainer or authorize an rjob. A-PES remains
ineligible until the normal PES/shuffled 512-row decisions trigger it.

Storage remains approximately 34G under `Faro_ailab`, below the 700G limit.

## Continuation checkpoint (20:43 HKT)

The PES retry loop was restarted under its existing lock after validating that
the old `runner_started` marker had no live owner.  The control plane still
returns DNS failure, so no rjob name or GPU allocation has been created.

The literature audit was reconciled with the actual sampler: PES evidence is
`max(candidate_logits) - candidate_logits[sampled_code]`, detached and based on
the sampled calibrated-support action.  A static regression test now rejects
accidental reintroduction of a top-1/top-2 proxy.  The low-GPU action-margin
diagnostic was also corrected to use its dedicated positive-tag allowlist on
the 16-GPU partition, preserving the 8 -> 6 -> 4 -> 2 -> 1 evaluation ladder.

Focused PES, A-PES, monitor, and tag-contract tests pass (`20 passed`).  The
full suite has two pre-existing import errors in PixVL-only tests because the
local environment has `huggingface-hub==1.21.0` while installed Transformers
requires `<1.0`; this does not affect the SAMTok/PES tests.  No threshold,
holdout result, or candidate promotion was changed.

An additional static regression in the closed R34 helper was fixed locally:
its smooth dominance score now uses a monotone softplus geometry transform, so
first-depth decay cannot reverse the ordering of otherwise stronger joint
improvements.  This does not reopen R34 or authorize a new training job.

The PES finalizer was tightened before any PES artifact existed: its canonical
`promotion_gate` now equals the CI-corrected utility/geometry/null gate for
both R18 and matched-SFT comparisons.  The historical legacy gate is retained
as an audit field only and cannot promote a candidate without a positive
utility confidence bound.

An end-to-end finalizer rehearsal using existing complete 512-row artifacts in
`/tmp` completed all five 20,000-bootstrap comparisons and emitted the expected
decision schema.  The rehearsal output was discarded and is not a PES result;
the real normal/shuffled inputs remain absent and therefore cannot be promoted.

## Continuation checkpoint (23:52 HKT)

The external lock-protected submitter remains the sole owner of the PES
transition.  Its latest retry at `23:50:07+08:00` still reports unresolved
`h.pjlab.org.cn`; `submitted`, normal metrics, both PES holdouts, and the
final decision are all absent.  The late monitor continues to heartbeat, with
the older candidates marked finished/submitted or waiting according to their
existing artifacts; it does not launch any PES control before the normal worker
gates pass.

The complete non-PixVL-dependent regression suite now passes (`196 passed`),
including the corrected visual-adapter trainability contract and R24 config
isolation.  All shell submitters pass `bash -n`, and the CPU A-PES probe still
reports the fixed `[0, 1, 2, 1]` states and seed-1907 shuffled control.  The
only omitted tests import legacy PixVL modules under an incompatible local
`huggingface-hub==1.21.0`; no training or evaluation code relies on those
imports.  Storage remains `34G` outputs plus `4.4G` eval artifacts, well below
the 700G cap.

## Branch safety checkpoint (23:56 HKT)

The heartbeat log showed older late-monitor instances still attempting to mark
the already closed paired-view branch as `eval_submitted`.  The shared
`submit_samtok_standalone_eval_adaptive.sh` entrypoint now has a final
fail-closed guard for the closed PV, BA, BS, and AB branches, and requires all
normal PES worker gates before a shuffled-PES evaluation can be submitted.
This protects the single-candidate protocol even if a stale monitor survives a
restart.  The new guard and all focused contracts pass (`25 passed`).

The same shared entrypoint now serializes every eval submit by output path and
reserves a five-minute retry slot.  This suppresses duplicate 8-GPU requests
from stale monitor instances while preserving the requested 8 -> 6 -> 4 -> 2
-> 1 ladder when a real submission succeeds.

The PV guard is fail-closed on a missing or non-`open` training decision, and
the shuffled-PES guard is fail-closed until a real normal PES worker artifact
passes all validity gates.  Runtime markers for the completed BA/BS/AB branches
were set to `CLOSED_BRANCH`; no PES marker was touched.

## Continuation checkpoint (2026-08-30 00:05 HKT)

The control plane remains unavailable (`h.pjlab.org.cn` DNS resolution fails)
and the PES submitter has no `submitted` marker or worker metrics.  Existing
monitor heartbeats are still active; alternating legacy heartbeats only report
marker state and cannot bypass the shared eval guard.  Current storage is
approximately `34G` outputs, `4.4G` eval files, and `35M` logs.

After the branch-safety changes, the full non-PixVL-dependent suite passes
(`197 passed`), all shell scripts compile with `bash -n`, and real
closed-branch probes reject PV/BA/BS/AB and shuffled PES before any rjob call.
