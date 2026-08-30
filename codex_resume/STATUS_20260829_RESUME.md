# FARO continuation status (2026-08-29)

## Active decision

FEPO-R18 remains the provisional holdout-selected SAMTok reference. PV-FEPO
completed the required 5,120-row/10-step/K=4 training contract but was closed
by its preregistered joint-positive support gate (`mean=0.10625 < 0.20`), with
no holdout quality claim. The matched continued-SFT control has complete
512-row/20k paired files and remains a control because boundary-hard/thin
boundary-IoU non-inferiority is not established.

## Next experiments

1. Submit exactly one AB-FEPO action-budget job after the `ailab-dnacoding`
   control plane is reachable. R35, BA-FEPO, and BS-FEPO have already closed;
   BS was rejected by its complete holdout/bootstrap decision.
2. Require all training validity gates before AB's complete 512-row/20k
   evaluation. Evaluation uses the fixed 8 -> 6 -> 4 -> 2 -> 1 GPU ladder,
   waiting 300 seconds at each nonterminal level; the 1-GPU job is terminal.
3. If AB is rejected, close the fixed-budget branch and design the next
   isolated hypothesis from the ten-paper synthesis. Do not tune thresholds
   from holdout outcomes or submit candidates concurrently.

## Control-plane status

As of this snapshot, `h.pjlab.org.cn` does not resolve on the login node. The
R35 submit attempt therefore failed before rjob creation; no task, checkpoint,
or evaluator output was created. The submitter and late-screen monitor are
idempotent and retry when invoked, with namespace `ailab-dnacoding`, `dna-`
names, and every positive tag in `rjob_tags.txt`. The R35 submitter now uses a
lightweight namespace probe before its heavyweight submit wrapper, preventing
repeated packaging attempts during DNS outages.

The shared SAMTok TB-GPPO and ES-GR-CPPO submit entry points also reject data
files below 5,000 rows and one-step job/config names. Historical smoke configs
remain in the repository for provenance, but cannot be submitted accidentally
under the current minimum 5,120-row/10-step policy.

The legacy `submit_samtok_sft.sh` entry point is explicitly disabled because
it invokes the PixVL trainer. Current SFT/FEPO work must use the standalone
SAMTok paths; PixVL remains limited to existing evaluation and data
interfaces.

All active SAMTok training submitters now call
`tools/validate_training_budget.py` before `rjob submit`. The validator loads
the actual config, counts nonempty JSONL rows, and rejects either configured
or actual data below 5,000 rows or `optimizer.max_steps < 10`. The registered
AB config passes with 5,120 rows and 10 steps. AB training has completed as
`dna-fepo-action-budget-native-rank-local-10s-77de4` and passed the worker
gates. Its evaluation submitter is retrying after a control-plane DNS
failure; no holdout result exists yet.

## Resource and provenance constraints

All new outputs are under `Faro_ailab`; no new files are written under
`PixVL_ailab`. Current usage is about 38G of the 700G limit. PixVL is used only
through the existing evaluator/data interface; active training is ordinary
SAMTok grouped RL, with no PixVL trainer, self-supervised cycle, OPD teacher,
router, expert, EMA, or inference-time routing.

### PES-FEPO continuation (2026-08-29)

PES is now wired into the SAMTok-only trainer with detached per-depth behavior
log-probabilities, predicted evidence scope masks, and auditable state/length
metrics. Registered normal and shuffled-evidence control configs both validate
against the 5,120-row/10-step/K=4 contract. A lock-protected transition waits
for the completed AB rejection and retries the positive-tagged `dna-` submit
every 300 seconds; the dnacoding API is currently unreachable by DNS, so no
duplicate job has been created.
The offline candidate probe now passes for PES and the shuffled control, including
finite scoped PPO loss and deterministic state coverage checks. The live retry
session has recorded control-plane failures at 18:34, 18:39, 18:44, 18:49,
and 18:54 and remains
active.
An updated monitor instance is also active under `logs/screen_monitor_v5`,
where both PES candidates are explicitly visible as `waiting`.

The environment-backed targeted suite now reports `39 passed, 1 deselected`
for PES, TB-GPPO, and submission/monitor contracts; the broader suite has unrelated legacy
dependency/test failures (HF Hub version mismatch and older PixVL/representation
assertions) and is not used as PES evidence.

## PES local audit (2026-08-29 19:20 HKT)

The probe entry point was made self-contained: `tools/run_fepo_candidate_probe.py`
now adds the repository and `Sa2VA` roots to `sys.path`, so direct invocation
from the FARO root no longer depends on worker-only `PYTHONPATH` setup. The
normal PES probe passed with the registered 5,120-row/10-step/K=4 contract and
finite credit diagnostics. The PES static tests passed (`2 passed`), and
trainer/contract/config compilation plus `git diff --check` passed. The
environment-backed PES/TB-GPPO subset reports `31 passed, 1 legacy R34
failure`; that failure is the already-closed soft-native-dominance assertion,
not PES.

The control plane was rechecked at 19:17 HKT and still failed DNS resolution
for `h.pjlab.org.cn`; `logs/pes_submit/submit.log` recorded the 19:19 retry.
No PES or shuffled-control rjob exists, and no output/checkpoint was created.
The late monitor heartbeat continues to mark both candidates `waiting`.
Storage remains about 39G, with all new artifacts under Faro_ailab.

## PES contract cleanup (2026-08-29 19:32 HKT)

Before any PES rjob was created, a static audit found that the registered
`confident_mass` and `ambiguous_mass` fields were dead parameters: the
implementation classified evidence using only mean normalized entropy and
native top-1/top-2 logit margin, matching the preregistered hypothesis. The
dead fields and call arguments were removed from both normal and shuffled
configs and from the contract; `top_support_masses` remains an audited,
detached diagnostic tensor. No behavior or threshold was changed after job
creation because no job existed. Python byte-compilation, shell syntax, and
`git diff --check` pass. The worker-only torch probe remains pending.

The PES submitter and late-screen monitor are running with lock protection.
The latest control-plane retry at 19:59 HKT still failed DNS resolution; no
PES or shuffled rjob exists and no storage was added.

The monitor continued emitting waiting heartbeats through 20:02 HKT; all
previous candidate holdouts remain finished or closed, and no PES output
directory or submitted marker has appeared.

The shuffled-control handoff was tightened before any job creation: normal PES
must finish with worker validity, effective-support, tail-risk, and PES-
coverage gates all passing. A normal worker failure now closes the branch
without launching a non-interpretable shuffled control. The corresponding
non-Torch static monitor suite passes 7 tests.

Conditional post-PES variants are documented in
`PES_NEXT_VARIANTS_20260829.md`: mass-aware evidence, boundary-first scope, and
null-safe scope. They are mutually exclusive and trigger-gated; none is
submitted or treated as a result while PES is pending.

## PES sampled-evidence correction (2026-08-29 19:47 HKT)

An audit found the rollout had been recording native top-1 versus top-2
candidate logits rather than the preregistered native-versus-sampled margin.
Before any rjob existed, it was corrected to
`max(candidate_logits) - candidate_logits[sampled_code]`; the sampled code is
the calibrated-support action used by that rollout. The helper now requires an
explicit margin tensor, and the offline probe/static checks pass it explicitly.
No threshold, reward, data, or holdout result was changed.

The ten-paper synthesis was also tightened: V-Zero's contrastive clean/negative
evidence, teacher replay, and the 2608.14144 line's EMA clean-teacher view
asymmetry are recorded as excluded mechanisms and inspiration only. PES makes
no teacher, OPD, EMA, view-distillation, or cycle claim.

The PES coverage counter was corrected before submission as well: each worker
now accumulates local evidence-state counts and the final reduce creates the
global count, avoiding duplicate absolute counts from gather-then-reduce.
