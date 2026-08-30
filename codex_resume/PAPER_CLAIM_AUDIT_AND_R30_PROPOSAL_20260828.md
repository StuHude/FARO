# FEPO Claim Audit and R30 Proposal (2026-08-28)

This audit checks the ten local paper records against
`LITERATURE_TO_FEPO_HYPOTHESES_20260828.md`, `FEPO_FINAL_METHOD_20260828.md`,
`EG_FEPO_PAPER_DRAFT.md`, and `RESULTS_LEDGER_20260828.md`. It is a claim
boundary, not evidence that a pending queue job has finished. No rjob is
submitted by this note.

## Executive audit

The completed evidence supports a narrower statement than a ten-paper-derived
unified theory:

> On the fixed SAMTok continued-SFT anchor, a single mask-or-null policy using
> native-relative joint cIoU/boundary-IoU credit and a first-divergence code
> scope produced a positive 512-row in-domain result at seed 17. The positive
> result is replicated across seeds, but the shuffled-depth control means that
> the *earliest depth itself* is not shown to be causal. RefCOCO transfer is
> non-significantly lower, so no general transfer improvement is established.

The literature is useful for generating tests (geometry slices, representation
interfaces, null outcomes, and reward calibration), but it does not jointly
validate FEPO. Most papers use different architectures, labels, decoders, or
training objectives. The final paper should use ``motivates a falsifiable
diagnostic`` rather than ``establishes`` or ``proves`` for these links.

## Ten-paper mapping audit

The local records are `../papers/text/*.txt` (with the three newer records in
`../papers/*.txt`). The exact title matters when naming a baseline.

| Paper record | What the paper actually establishes | Current FEPO mapping | Audit / required correction |
| --- | --- | --- | --- |
| Qwen3-VL-Seg, arXiv:2605.07141 | A 17M box-guided mask decoder; multi-scale features, spatial-semantic queries, high-resolution box gating, and iterative mask-aware refinement. It builds SA1B-ORS and ORS-Bench with ID plus six OOD shifts. | Positive/negative/difficult slices and boundary-sensitive evaluation. | The paper does not establish an RL reward or a three-way target-present/negative/difficult training split. It mentions multi-target/no-target through the GRefCOCO context, but that is not evidence for FEPO's sentinel constraint. Say “motivates OOD, scale, occlusion, and boundary slices”; do not say it validates null calibration. |
| PixVL | A discrete pixel-token interface and a cyclic/self-supervised training line in its own system. | Pixel trajectories are directly scoreable; PixVL training is excluded. | This exclusion is correct. “Routing” is not a paper-supported mechanism to import into FEPO; call the prior system's branches/cycle what they are and state that no PixVL weight, verifier, OPD target, or cycle is used. |
| EVP, arXiv:2312.08548 | Enhanced Visual Perception for depth estimation: inverse multi-attentive feature refinement over visual/diffusion features, evaluated on NYU Depth v2 and KITTI. | Visual representation and boundary/detail quality can bottleneck segmentation. | EVP is not a segmentation experiment and does not prove a segmentation boundary bottleneck. Use it only as an analogy that dense prediction may be interface/feature limited. Replace “EVP shows segmentation boundary quality bottlenecks” with “EVP motivates testing visual-feature refinement as a representation hypothesis.” |
| SenseNova-U1, arXiv:2605.12500 | A native unified understanding/generation model with a near-lossless visual interface, joint objectives, and native MoT; it reports staged data/training and broad capability evaluations. | Staged post-adaptation checks and capability retention. | The local paper is titled **SenseNova-U1**, not “SenseNova-Vision.” Staging and broad evaluation motivate a protocol, not a FEPO mechanism or proof of retention. Record the title/ID explicitly and do not claim that SenseNova validates a null-risk sentinel. If the intended “SenseNova-Vision” is another paper, it must be separately located and read. |
| Fine-R1, arXiv:2602.07605 | CoT SFT followed by TAPO: intra-class and inter-class triplet image augmentation plus DAPO-style token-level policy optimization for fine-grained recognition. | Stabilize with SFT before grouped relative RL. | Only the ordering and motivation for relative on-policy optimization are borrowed. FEPO does not inherit TAPO, triplets, category labels, or its generalization evidence. K=4, first-divergence scope, and geometry credit remain FEPO hypotheses, not Fine-R1 findings. |
| Dr. Seg, arXiv:2603.00152 | A GRPO perception framework using Look-to-Confirm reasoning and Distribution-Ranked Reward for box/point outputs driven into SAM2; its reward coordinates include box/count/point metrics. | Rank-based continuous geometry feedback and broader perception exploration. | The exact title is **Dr. Seg**, not “DR²Seg” in the local PDF. It does not use native SAMTok code, cIoU+boundary-IoU, or a null sentinel. R21/R22 may be described as an adaptation test of reward ranking; no direct transfer of its causal claims is justified. |
| Latent Denoising, arXiv:2604.21343 | Training-time saliency-aware masking/Gaussian corruption of projected visual tokens, supervised recovery of clean teacher patch features, and intra-image contrastive structure preservation; corruption is disabled at inference. | Latent/view perturbation motivates representation robustness diagnostics. | This is not view asymmetry or OPD; S2VOPD is the view-asymmetry paper. Replace “latent/view perturbation” with “training-time latent corruption/recovery.” Do not imply that FEPO's R19 view-drop is a replication. |
| OpenWorldSAM, arXiv:2507.05427 | Frozen SAM2 and BEiT-3 with a roughly 4.5M language adapter, positional tie-breakers, and language/image cross-attention for open-vocabulary multi-instance masks; optional second-pass refinement mostly changes visual quality. | Explicit absence and ambiguity outcomes; small interface adaptation. | OpenWorldSAM does not define FEPO's no-target sentinel or a lower-tail risk constraint. It motivates recording absence/ambiguity and testing a lightweight interface, but no OpenWorldSAM decoder, matching, or checkpoint enters training. |
| V-Zero, arXiv:2606.25319 | OPD with a teacher-side task-relevant crop and equal-size distractor crop. A detached contrastive evidence gap gates dense positive-view distillation; the target remains the positive teacher view. | Detached uncertainty/evidence can downweight fragile trajectories. | Evidence gap is not generic uncertainty, and V-Zero is OPD rather than RL. R19's same-policy view-drop proxy has no teacher positive/negative crop and therefore must remain a negative control, not a V-Zero implementation. R25's entropy/missing-support multiplier is a new FEPO hypothesis, not a result from V-Zero. |
| S2VOPD, arXiv:2608.14144 | EMA teacher sees the clean image; student samples on a moderately degraded view; top-k generalized JSD supplies self-supervised OPD. Moderate information loss helps, while aggressive crops can remove task evidence. | Moderate perturbation is a robustness diagnostic; S2VOPD training is excluded. | This boundary is correct. S2VOPD cannot be cited as support for a ground-truth cIoU RL credit rule. Any augmentation experiment must report whether the target remains answerable and cannot become a hidden teacher/cycle. |

### Cross-paper identity and causality warnings

1. `DrSeg_2603.00152.pdf` is the local source for **Dr. Seg**. Use the exact
   title in tables and citations. The user's “DR²Seg” wording should not turn
   the local paper into a different method without a separate source.
2. `SenseNova_2605.12500.pdf` is **SenseNova-U1**. A claim about
   “SenseNova-Vision” requires another paper record; otherwise the current
   mapping is not auditable.
3. `LatentDenoising_2604.21343` and `S2VOPD_2608.14144` are distinct. The
   former has supervised latent feature recovery; the latter creates
   teacher/student view asymmetry. They must not be merged as one perturbation
   result.
4. No paper above establishes that native-relative reward, code-depth
   localization, or a fixed no-target sentinel is individually causal. Those
   are FEPO components to test with matched ablations.

## FEPO document and result claims

### Supported as written, with scope attached

- R18 has a complete 512-row paired holdout and 20,000 paired bootstrap. The
  reported positive cIoU delta `+0.023769` with CI
  `[+0.008231,+0.040535]` and utility delta `+0.090010` with CI
  `[+0.066087,+0.115072]` support an in-domain comparison to the named frozen
  anchor.
- The R16/R18 direct comparison is compatible with cross-seed non-inferiority
  under the registered margin. It supports replication, not independence of
  every component.
- R19 is a failed/negative evidence-gating control. It must not be used as a
  positive V-Zero claim.
- The GRefCOCO report is complete and may be reported with its dataset size and
  point estimates. The RefCOCO point estimates are lower with intervals crossing
  zero; the draft correctly labels this as a non-significant transfer
  regression.
- R21--R28 are pending screens until a valid adapter and metrics file exist.
  Queue state is not quality evidence.

### Claims that must be narrowed in the final paper

- “First-divergence localization improves learning” is too strong. R15's
  shuffled-depth control also passes, so the supported term is **a local credit
  scope aligned with the mask-token interface**. Report earliest depth as a
  registered scope choice, not a proven causal location.
- “The ten papers support FEPO” is too broad. The defensible statement is that
  ten papers supplied independent hypotheses or evaluation controls; most were
  not architecture-compatible and several training objectives were explicitly
  excluded.
- “EVP shows the frozen visual merger is the bottleneck” is unsupported. The
  completed projector-plasticity evidence is needed for that statement, and
  the paper itself is depth estimation. At most, R18's transfer limitation and
  prior visual-adapter probes motivate a representation-interface test.
- “Uncertainty-calibrated” should not be used as a V-Zero result label. V-Zero
  supplies a contrastive evidence gate; FEPO's entropy/support confidence is a
  separate, detached heuristic whose value is unproven until R25 passes.
- “Generalization” should be reserved for a complete, pre-registered transfer
  comparison. The current RefCOCO result does not support a transfer-gain
  claim, and no OOD benchmark from Qwen3-VL-Seg or OpenWorldSAM has been run
  under the SAMTok-only contract.
- R18's high holdout gain does not establish superiority over Qwen3-VL-Seg,
  Dr. Seg, OpenWorldSAM, or V-Zero because those systems use different output
  spaces and/or training objectives and are not valid SAMTok-only controls.

## R30: supervised dual-view grounded-interface FEPO

This is one conditional candidate for a future screen. It is intentionally
not submitted while R21--R28 occupy the queue.

### Hypothesis

R18 changes mask-token policy probabilities but does not directly test whether
the visual-to-language interface can preserve target evidence under ordinary
input variation. A small, trainable visual-merger adapter, trained with the
ground-truth SAMTok code on a mild target-preserving second view, may reduce
the in-domain/RefCOCO mismatch. The test is **supervised augmentation plus
R18 RL**, not OPD, teacher distillation, counterfactual labeling, or a
PixVL-style cycle.

The hypothesis is rejected if the visual adapter has no verified gradient or
logit effect, if the clean holdout geometry is non-inferior but no better than
the matched visual-SFT control, or if null/format/slice gates regress.

### Fixed minimal screen

- Initialization: the exact `continued_sft_to500` SAMTok adapter and seed 17
  used for R18; no PixVL checkpoint or weight path.
- Data: exactly the existing `egfepo_train_5120.jsonl`, 5,120 rows with the
  same train/holdout image-disjoint provenance. No extra data are introduced.
- Optimization: 10 outer steps, K=4 grammar-valid siblings, the existing
  effective-support controller, and the R18 native-relative joint geometry
  credit (`minimum_improvement=1e-4`, depth decay `0.85`). Keep the shared
  32-row no-target sentinel and every existing finite-ratio/validity/null-risk
  gate.
- Trainable parameters: the same language-policy LoRA as R18 plus only the
  named SAMTok visual merger/deepstack-merger LoRA targets at fixed rank 16.
  The base model and SAMTok decoder remain frozen. The exact trainable-key
  allowlist and parameter count must be written to provenance before training.
- Auxiliary loss: on every paired row, apply one fixed mild photometric,
  geometry-preserving image transform (brightness `1.03`, contrast `0.97`) and
  apply ground-truth SAMTok mask-code cross-entropy on that same row's view.
  The mask code is reused unchanged because no pixel geometry moves; on
  no-target rows the canonical null CE is used. Add this replay with fixed
  `lambda_sup=0.10` to the R18 objective; no coefficient sweep is allowed.
  The clean-view RL rollout remains the only source of geometry reward.
- Controls: (a) R18 language-only FEPO at matched rows/steps/rollouts; (b) a
  visual-merger-only supervised replay with the same transformed views and
  optimizer budget. The latter tests whether a gain is ordinary interface SFT
  rather than an RL interaction. A clean-view visual-SFT control can be added
  only if queue capacity permits, with its data/steps fixed in advance.

### Non-negotiable no-op and leakage checks

The earlier visual adapter history includes a silent adapter-composition risk.
Before promotion, a worker-side preflight must assert all of the following:

1. At least 80% of optimizer steps have finite, nonzero visual-merger gradient
   norm above the pre-registered floor `1e-8`.
2. Merged visual-adapter logits differ from the frozen anchor on a fixed
   training probe (record max absolute logit delta and changed-token count).
3. The output adapter contains only the allowlisted visual-merger and language
   LoRA keys; no external visual encoder, PixVL module, teacher, or decoder is
   present.
4. The transformed-view mask code is exactly the row's ground-truth SAMTok
   mask code because the registered transform is photometric only; no holdout
   mask or evaluator output is read during training.

Failure of any preflight closes R30 as an implementation failure and does not
justify changing rank, learning rate, transform strength, or loss weight.

### Promotion gates

First run the standard complete 512-row clean holdout and 20,000 paired
bootstrap. Relative to R18, require:

- positive cIoU mean improvement at least `+0.005` with a paired 95% CI lower
  bound above zero;
- utility CI lower bound at least `-0.010`, no-target-recall CI lower bound at
  least `-0.010`, invalid output rate `0`, and positive-mask rate at least
  `0.95`;
- no fixed small/thin/boundary/area slice drops by more than `0.010`;
- the candidate beats the visual-SFT-only control on positive cIoU, or is
  explicitly reported as interface SFT rather than an RL interaction.

Only a screen passing these gates can receive complete RefCOCO/GRefCOCO
evaluation. The official evaluation is not used to tune `lambda_sup` or the
transform. A transfer regression remains a result, not a reason to alter the
recipe.

### Relation to the ten papers

R30 borrows only testable lessons: Qwen3-VL-Seg's emphasis on fine spatial
detail and scale, OpenWorldSAM's small trainable language/visual interface,
EVP's general dense-feature refinement motivation, and Latent Denoising's
training-only perturbation/robustness framing. It does **not** copy their
decoders, pretrained visual features, Hungarian matching, latent teacher
targets, OPD, EMA teacher, or S2VOPD/V-Zero view-distillation objectives.

## Queue and resource boundary

This note does not submit R30 or modify the eight existing jobs. If later
approved, it must use a `dna-` name, namespace `ailab-dnacoding`, positive tags
from `rjob_tags.txt`, and at most two GPUs for the 5,120-row/10-step screen.
All artifacts stay under `Faro_ailab`; evaluation keeps the complete 512-row
protocol and the `8 -> 6 -> 4 -> 2 -> 1` five-minute fallback. Storage and GPU
budgets remain unchanged.
