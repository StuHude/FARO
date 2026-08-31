# FARO: Failure-Evidence-Conditioned Pixel RL

FARO is a SAMTok-only study of grouped policy optimization for referring
segmentation.  The current method, FEPO, uses one SAMTok policy, a fixed
mask-or-null grammar, native-relative cIoU/boundary-IoU reward, and detached
rollout evidence to choose which changed mask-code depths receive the PPO
credit.  It does not use a PixVL trainer, PixVL checkpoint, OPD teacher, EMA
cycle, counterfactual labels, inference router, or second expert.

## Evidence status

R18 is the promoted provisional reference: native-anchored rank-local joint
geometry credit at the first changed SAMTok depth.  Its complete paired
holdout contains 512 image-disjoint rows (256 positive and 256 no-target) and
20,000 paired bootstrap repetitions.  The current open hypothesis is
PES-FEPO (predicted-evidence scope), which keeps the R18 reward and uses
detached entropy and sampled-action margin to select a one-depth, two-depth,
or empty update scope.  Its deterministic shuffled-evidence control is
submitted only after the normal worker passes every validity gate.

The authoritative experiment decisions are recorded in:

- `RESULTS_LEDGER_20260828.md`
- `FINAL_RESULTS_TABLE_20260831.md`
- `FEPO_CONTINUATION_DECISION_TABLE_20260828.md`
- `LITERATURE_CLAIM_AUDIT_20260829.md`
- `PES_FEPO_PREREG_20260829.md`
- `GOAL_COMPLETION_AUDIT_20260831.md`
- `codex_resume/STATUS_20260831_CONTINUATION.md`

Offline probes and static tests are implementation evidence only; they are
not model-quality results.  Until the normal PES rjob finishes and produces a
valid worker manifest, the overall goal remains open and R18 remains the only
defensible quality claim.

## Repository layout

- `Sa2VA/projects/samtok_selective/`: SAMTok data, grammar, trainers,
  geometry credit, PES scope logic, configs, and worker contracts.
- `Sa2VA/projects/samtok_selective/configs/`: preregistered SFT, FEPO, PES,
  controls, and evaluation configurations.
- `scripts/`: dnacoding submission wrappers, adaptive evaluation/finalizers,
  local probes, and audit helpers.
- `tools/`: deterministic contract probes, budget validation, evidence and
  evaluation utilities.
- `tests/`: static and CPU contract tests for the training/evaluation gates.
- Root Markdown files and `codex_resume/`: literature, preregistration,
  decision, result, and continuation records.

Datasets, checkpoints, caches, logs, and evaluation outputs are intentionally
ignored by git.  They stay under the local `Faro_ailab` workspace and are not
part of the GitHub repository.

## Reproducibility contract

Every training submission must satisfy all of the following:

- namespace `ailab-dnacoding` and a job name beginning with `dna-`;
- every positive tag listed in `rjob_tags.txt`;
- at most 24 GPUs and less than 700G workspace storage;
- at least 5,120 training rows, at least 10 optimizer steps, and K=4 rollouts;
- SAMTok initialization only, with no holdout access during training;
- complete 512-row evaluation and 20,000 paired bootstrap repetitions.

Evaluation jobs use the required adaptive GPU ladder `8 -> 6 -> 4 -> 2 -> 1`,
waiting 300 seconds before each downgrade; the one-GPU job is left queued.

## Local verification

Run from the repository root with the project environment active:

```bash
PYTHONPATH=third_party/transformers:Sa2VA:. pytest -q tests/test_*static.py
python tools/run_fepo_candidate_probe.py
python tools/run_apes_contract_probe.py
```

The standalone budget audit is fail-closed and requires an explicitly
approved SAMTok base checkpoint:

```bash
SAMTOK_BASE_CHECKPOINT=/path/to/samtok/checkpoint \
  python tools/validate_training_budget.py
```

## Training and evaluation entry points

The registered PES stages are launched through:

```bash
bash scripts/submit_samtok_tb_gppo_predicted_evidence_scope.sh
bash scripts/submit_samtok_tb_gppo_predicted_evidence_scope_shuffled.sh
```

The shuffled control must not be launched before the normal PES completion
and worker-gate marker.  Adaptive standalone evaluation and the complete
holdout finalizer are available through:

```bash
bash scripts/submit_samtok_standalone_eval_adaptive.sh
bash scripts/finalize_pes_eval.sh
```

All scripts perform their own path, budget, namespace, positive-tag, and
provenance checks.  A failed worker validity gate is recorded as failure and
does not authorize holdout evaluation or promotion.

## Literature boundary

The ten archived papers (Qwen3VL-Seg, PixVL, EVP, SenseNova-Vision,
Fine-R1, DR2Seg, Latent Denoising, OpenWorldSAM, V-Zero, and S2VOPD) provide
hypotheses and evaluation discipline only.  The traceable mechanism/result
separation is maintained in `LITERATURE_CLAIM_AUDIT_20260829.md`.
