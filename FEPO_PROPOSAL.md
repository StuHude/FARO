# FEPO: Failure-Evidence-Conditioned Pixel Policy Optimization

## Decision

The first broad FEPO candidate is **not promoted**. The fixed balanced
positive/no-target calibration gate (2026-08-18) gives shared17 `0.6582`
balanced accuracy versus relation17 `0.6543`; no-target abstention recall is
`0.3164` versus `0.3086`. See `../FEPO_CALIBRATION_GATE_20260818.md`. The
relation-margin development cIoU gain therefore remains a diagnostic result,
not a paper claim.

The current Idea3 route is a useful diagnostic, but it is not yet a unified
training method.  `atom_conditioned` extracts text atoms and mask metadata,
chooses a hard bucket, and changes reward/loss scales.  OPD is still one shared
answer-span JSD.  The result is reward reweighting, not three correction
policies; geometry metadata can also be a training-only oracle.

The proposed replacement remains a useful framework boundary, but the next
testable version is **NC-FEPO** (null-calibrated FEPO): add an
image-conditioned null-vs-target verifier before using failure evidence for
local credit. The lexical/confidence-only probe is insufficient for opaque
refseg mask tokens. Concrete follow-up candidates and rejection criteria are
recorded in `../FEPO_NEXT_IDEAS_20260818.md`.

## Baseline lineage contract

Every trainable arm and baseline starts from the original SAMTok checkpoint.
PixVL is allowed only as a source of evaluation code and data. No PixVL
checkpoint, verifier, learned score, training recipe, or pre-trained adapter may
initialize or supervise a FARO arm. Teacher, reference, and student adapter
paths are `None` at initialization; matched arms differ only in the
pre-registered FARO objective.

## Method

For every rollout, compute three deficits from signals available without the
ground-truth answer:

- semantic: missing evidence coverage plus unsupported-claim/no-target rate;
- relation: target probability minus the strongest same-image or cross-image
  confuser probability;
- geometry: boundary disagreement and area error of the predicted mask.

Apply a temperature-controlled softmax to these deficits.  Correct samples use
the ordinary policy objective.  Failed samples receive local correction in all
active capabilities:

```text
L = L_policy(correct) + 1[failure] * sum_k w_k L_local,k
    + lambda_preserve L_T2M/general + beta L_KL
```

`L_local,k` is scoped to `semantic_text`, `relation_and_mask`, or `mask` tokens
through the existing task-matched scope utility.  A privileged GT/overlay
teacher can scale correction magnitude, but cannot select the route or change
the update sign.  A short general multimodal/caption calibration stage follows
segmentation-heavy training.

## Falsifiable experiment ladder

1. **Offline deterministic smoke (no GPU):** verify evidence bounds, soft-route
   normalization, failure gating, and task scopes on fixed synthetic records.
2. **Matched-data 200-step GPU smoke:** compare unified objective, hard bucket,
   FEPO `soft_local` route, and shuffled-route control with identical examples,
   groups, updates, and decoding in one 8-GPU job.
3. **Ablations:** remove confusers, boundary term, and unsupported-claim/no-target
   penalty independently; compare predicted-only router to GT/oracle upper bound.
4. **Calibration gate:** report DLC recognition/positive/negative separately,
   GT-mask-to-caption versus predicted-mask-to-caption, and low-temperature /
   short-decoding controls.
5. **Scale only after a positive smoke:** nested SAV-first 10K/50K/200K runs;
   report RefCOCO family, DLC, LVIS/PACO, choose-one accuracy, shortcut rate,
   and worst-slice delta at the same checkpoint.

The method is rejected if it fails either two pre-registered slices or the
shuffled/matched-compute control.  A positive aggregate score with degraded
DLC negatives is not a success.

## Literature boundary

Qwen3-VL-Seg motivates explicit category/descriptive splits and a post-segmentation
general-capability stage.  PixVL already owns cross-view cycles, hard confusers,
choose-one, and mask-token credit; FEPO must not claim those as novelty.  EVP,
SenseNova-Vision, Fine-R1, DR2Seg, and arXiv:2604.21343 still require source
verification in this offline workspace before exact related-work claims.

## Resource contract

All future jobs must use `namespace=ailab-dnacoding`, `charged-group=dnacoding_gpu`,
the shared dnacoding mount, and a `dna-` job name.  A submission script must
refuse a name without that prefix and must be run only after a live-GPU snapshot;
the project cap is 24 GPUs and 700G additional storage.

## First smoke status (2026-08-16)

- local CPU smoke: passed (`FEPO_SMOKE_OK`);
- rollout-time predicted-only adapter is implemented behind
  `routing.mode=predicted_only_evidence`; legacy `source_bucket` remains the
  default so existing runs are unchanged;
- the routed trainer no longer overwrites `self_privileged_rollout` teacher
  logits with a second unconditional overlay pass;
- `dna-fepo-contract-smoke-33132083`: failed before Python startup because the
  image environment was not unpacked (execution/infrastructure failure);
- `dna-fepo-contract-smoke-r1-52846616`: succeeded after adding the standard
  `/opt/vlm_env.tar.gz` bootstrap and emitted `FEPO_SMOKE_OK` on dnacoding.

No model-training claim is made from this contract smoke.  The next valid gate
is the matched-data 200-step experiment after a runnable routed schema and
checkpoint lineage are selected. The historical configs still contain
`/mnt/pfs/xiaoyicheng` defaults; set `FARO_PROJECT_ROOT`, `FARO_WORKSPACE_ROOT`,
`FARO_SAMTOK_MODEL`, and `FARO_BASE_CONFIG` in the dnacoding job or create a
pathized experiment config before submitting the GPU smoke.

## Rollout and schema evidence (2026-08-16)

- `dna-fepo-rollout-smoke-r2-58921054` succeeded on an H200. The 4B SAMTok
  model emitted valid mask tokens and the deployment route used only prompt,
  rollout text, and confidence.
- `dna-fepo-evidence-sweep-r1-28824740` exposed two issues: opaque refseg
  outputs need a confidence-based relation probe, and explicit no-target
  prompts must be treated as semantic abstention failures.
- The predicted-only adapter now handles those cases and keeps GT fields out
  of the route. A real-model rerun, `dna-fepo-evidence-sweep-r2-40187727`,
  completed successfully; semantic and relation caption probes route to their
  corresponding local credit, while ordinary refseg routes to geometry or a
  relation/geometry soft mixture.
- `dna-fepo-schema-smoke-r1-41035011` decoded 16 current PixVL refseg and 16
  maskcap records into `data/pixvl_idea3/schemas`, proving that the current
  PixVL self-supervision format can feed the Idea3 unified dataset contract.

These are contract and routing-evidence results, not a quality claim. The
first quality gate is the completed matched-data run using exactly those 32
records and the same checkpoint for all route arms. Any new quality run is
submitted as a single `dna-` job requesting 8--24 GPUs; no 2-GPU jobs are
allowed under the current resource contract.
