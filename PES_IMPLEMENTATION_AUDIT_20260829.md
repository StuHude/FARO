# PES-FEPO implementation audit (2026-08-29)

## Checks

- `fepo_gr_cppo_trainer.py`, `tail_gppo_contract.py`, and both PES configs pass
  Python byte-compilation.
- The dynamic probe could not run in this control shell because its Python
  environment has no `torch` module. It must be rerun in the SAMTok worker
  image before accepting a training artifact.
- No rjob was submitted by this audit.

## Finding and resolution: dead mass thresholds

`predicted_evidence_scope_masks` previously accepted `confident_mass` and
`ambiguous_mass`, although state assignment depended only on mean normalized
entropy and mean native margin. Because the preregistered hypothesis explicitly
uses entropy + margin, the dead mass keys were removed before any rjob was
created. `top_support_masses` remains an audited diagnostic tensor, but is not
used as a hidden routing threshold. PES is therefore unambiguously the
registered entropy+margin experiment.

## Follow-up resolution: sampled-relative margin

The initial audit also found that the rollout path was recording a native
top-1-versus-top-2 candidate margin, while the PES preregistration specified a
native-versus-sampled margin. Before any rjob was created, the rollout code was
corrected to record `max(candidate_logits) - candidate_logits[sampled_code]`.
The sampled code is the actual calibrated-support action selected by that
rollout, and the tensor remains detached under the existing `no_grad` block.
The probe and static test now pass this margin explicitly. This restores the
registered native-vs-sampled evidence semantics without changing thresholds or
reward credit.

The state is still reduced over mean per-depth entropy and margin, and scope
width is still one versus two changed depths; no claim of per-depth evidence
alignment is made beyond the registered scope rule.

The audit also found duplicated absolute state counts: the previous code
gathered states and accumulated the gathered count on every rank before the
final reduce. It now accumulates local counts and local observations, using the
gathered tensors only for step-level display. Final coverage fractions and
counts therefore represent one global population.

## Gate interpretation

The final coverage gate requires at least two observed states and at least 20%
for each non-empty state. State counts now accumulate locally and are reduced
exactly once, while gathered states are used only for step-level display. Both
fractions and absolute counts therefore describe one global population.

## Queue status

The PES submitter is still retrying every 300 seconds because `rjob list
--namespace=ailab-dnacoding` cannot resolve the control-plane host. No PES or
shuffled-control job marker exists. The latest observed retry is 19:54 HKT;
storage remains below the 700G limit.

The shuffled-control transition is validity-gated before submission: normal
PES must be `finished` with worker validity, effective-support, tail-risk, and
PES-coverage gates all true. A failed normal worker run closes the branch
without launching an uninterpretable control. The non-Torch monitor/static
suite passes 7 tests after this change.

## Current continuation addendum (2026-08-31)

A CPU dummy-model rollout with offset, non-contiguous code-token ids exposed a
sampler indexing defect in the native-vs-sampled evidence diagnostic: the
sampled global vocabulary id was being used to index depth-local candidate
logits.  The implementation now gathers with the depth-local sampled index;
this fix was made before any PES job or checkpoint existed.  The focused
PES/A-PES, submission-contract, and policy-isolation suite passes `28 passed`,
including effective-support sampling and action-term rescoring.

The normal PES retry remains lock-protected.  At the latest recheck
(`2026-08-31T21:48:49+08:00`), `rjob list --namespace=ailab-dnacoding` still
fails at unresolved proxy/DNS, the `submitted` marker is absent, and no PES
worker, holdout, or bootstrap result exists.
