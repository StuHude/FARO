# Literature claim audit for SAMTok-only FEPO (2026-08-29)

This appendix separates (a) mechanisms stated in the locally archived papers,
(b) the hypothesis imported into FEPO, and (c) evidence actually produced by
the FEPO experiment ledger.  A paper citation is not treated as evidence that
an FEPO arm works.  Section numbers below refer to the searchable local text
files under `../papers/` (the PDFs are retained beside them).

## Traceable claim matrix

| Paper and local source | Paper fact (traceable location) | FEPO interpretation/hypothesis | FEPO status and claim boundary |
| --- | --- | --- | --- |
| **Qwen3VL-Seg**, `papers/text/Qwen3VL_Seg_2605.07141.txt` | `3 Method`, `3.2.1-3.2.4`: a box-guided, multi-scale dense decoder with spatial-semantic queries and iterative refinement. `4.2.3 ORS-OOD-Bench` and `6.5 OOD Referring Segmentation`: explicit category/area/occlusion/lighting/instruction/risk-sensitive OOD evaluation. `5.1 Multi-stage Training`; `6.7 Ablation studies`: architecture and staged-training analyses. | Evaluate positive, no-target, thin/boundary and OOD slices, and check post-adaptation capability. Do not import its decoder, box path or SA1B-ORS curation. | **Partially verified protocol only.** FEPO's 512-row positive/null and geometry slices implement the evaluation discipline. No Qwen decoder/data claim is made; this paper does not by itself justify a null loss or a router. |
| **PixVL**, local repository/PixVL paper and `paper_survey_report.md` (full-text audit) | Cycle mask-to-text/text-to-mask, cross-view confusers, choose-one cold start, caption coupling, and mask-token-only credit are PixVL mechanisms. | Keep the mask-token interface and evaluator/data compatibility, but test one independent SAMTok policy; no cycle, verifier, teacher, OPD, EMA or PixVL checkpoint. | **Restriction verified by contracts/manifests.** PixVL mechanisms are exclusions/related work, not FEPO evidence. Any statement that FEPO uses PixVL's routing or self-supervision is unsupported. |
| **EVP**, `papers/text/EVP_2312.08548.txt`, `3 Methodology`, `4.5 Ablation Study` | Enhanced visual perception is an architecture/feature method: inverse multi-attentive feature refinement and regularized image-text alignment, with a Stable-Diffusion perception backbone; it is not a policy router. | Treat visual-interface quality as a separate control; a policy-credit gain that fails boundary tests may indicate a representation bottleneck. | **R35 was a separate visual-interface arm and failed its worker sentinel gate** (`RESULTS_LEDGER_20260828.md`, R35 section), so there is no EVP-derived quality claim. |
| **SenseNova-Vision / SenseNova-U1**, `papers/text/SenseNova_2605.12500.txt`, `3.1-3.5`, `3.4 Training Procedure`, `5.2 Ablation Studies` | Native unified multimodal architecture, near-lossless visual interface, joint objectives and staged training; broad understanding/generation evaluation. | Require capability-retention and transfer checks after segmentation specialization; do not call this a failure-router or pixel-RL method. | **Evaluation motivation only.** FEPO's official RefCOCO/GRefCOCO transfer is reported, including a documented RefCOCO regression; no SenseNova architecture or training objective is used. |
| **Fine-R1**, `papers/text/Fine_R1_2602.07605.txt`, `4.2 CoT SFT`, `4.3 Triplet Augmented Policy Optimization`, `5.3 Ablation Study` | TAPO uses anchor/positive/negative triplets, intra-class trajectory mixing and inter-class separation after CoT SFT. | Use a strong SAMTok SFT anchor and sibling-relative grouped policy optimization with complete support. | **Grouped-RL initialization is supported as a design analogy, not a TAPO reproduction.** FEPO uses K=4 geometry rollouts and no triplet/counterfactual text labels; no Fine-R1-specific gain is claimed. |
| **Dr. Seg / DR²Seg**, `papers/text/DrSeg_2603.00152.txt`, `4.2 Look-to-Confirm`, `4.3 Distribution-Ranked Accuracy Reward`, `5.4 Ablation Study` | Look-to-Confirm expands structured visual evidence in the output; each continuous accuracy component is mapped to a FIFO empirical quantile (queue size 2,048 in the paper) before aggregation. Ablations compare raw versus ranked rewards and queue sizes. | Use joint cIoU/boundary-IoU and rank controls to separate reward scale from credit scope. | **Rank/reward controls are implemented, but not Dr. Seg.** R21-R34 and the ledger reject their FEPO variants; FEPO must not claim Look-to-Confirm or Dr. Seg's reward as the source of R18's gain. |
| **Latent Denoising**, `papers/text/LatentDenoising_2604.21343.txt`, `3.1 Setup and Motivation`, `3.2 Latent Corruption & Recovery`, `4.2 Corruption Evaluation Protocol`, `4.5-4.6 analysis/ablations` | Training-only saliency-aware mask/Gaussian corruption of projected visual tokens, teacher patch-feature recovery and intra-image structure/contrastive losses; corruption heads are disabled at inference. | A robustness diagnostic can test whether FEPO depends on brittle visual features; this is orthogonal to credit routing. | **No denoising objective is implemented.** The literature note should not call this a generic view-perturbation or evidence-gating method; view-drop R19 belongs to S2VOPD/V-Zero-inspired controls and was rejected. |
| **OpenWorldSAM**, `papers/OpenWorldSAM_2507.05427.txt`, `3 Methodology`, `4.1-4.2 protocols`, `4.4 Ablation Studies` | Frozen SAM2/BEiT-3 encoders plus a small language adapter, positional tie-breakers and cross-attention soft prompting; open-vocabulary multi-instance and OOD/risk-sensitive segmentation. | Treat ambiguity, absence and OOD as evaluation/tail-risk concerns and protect no-target behavior while optimizing geometry. | **Only the tail-risk interpretation is FEPO's extension.** OpenWorldSAM does not establish a no-target abstention loss or sentinel constraint; R26/R29 and R35 results are FEPO experiments, not OpenWorldSAM evidence. |
| **V-Zero**, `papers/VZero_2606.25319.txt`, `On-Policy Distillation with Teacher-Side Views`, `Method`, `Contrastive Evidence Gating`, `Experiments/Ablation Study` | Student samples full-image trajectories; a teacher replays them using paired positive target crops and negative irrelevant crops, computes token log-probability gaps, then gates positive-view OPD. The signal is paired-view teacher evidence, not plain entropy or a native top-1/top-2 margin. | At most borrow the principle that a detached, predicted-only signal can gate already verified credit, while excluding teacher/OPD/view machinery under SAMTok-only constraints. | **R19 view-drop and R25 uncertainty screens were rejected.** The current PES entropy+margin scope is an independent hypothesis. It must not be described as implementing V-Zero contrastive evidence gating. |
| **S2VOPD**, `papers/S2VOPD_2608.14144.txt`, `3.1 Self-Supervised Visual On-policy Distillation`, `3.2 Constructing Asymmetry for Free`, `4.3-4.4 augmentation/ablation` | EMA teacher sees the clean image while the student rolls out on a degraded view; moderate information reduction (downscale/noise) is useful, while evidence-destroying crops are harmful. | A view-consistency stress test may diagnose representation brittleness, but cannot enter the core SAMTok-only RL objective. | **No S2VOPD objective is used.** R19/PV are controls only; PV was closed by its training support gate and no asymmetric-view claim is made. |

## Current PES correspondence audit

The PES preregistration says “native-vs-sampled logit margin”
(`PES_FEPO_PREREG_20260829.md`, Hypothesis).  The current sampler records
`max(candidate_logits) - candidate_logits[sampled_code]` in
`sample_effective_support_grammar_rollouts`.  The sampled code is the actual
calibrated-support action of that rollout, so this is action-aware evidence,
not a native top-1-vs-top-2 confidence proxy.  The tensor is detached under
the rollout `no_grad` context and is used only to select the fixed scope.
The state rule still averages entropy and this margin over depths before
choosing a one-depth or two-depth scope; no per-depth causal claim is made.

`top_support_masses` is passed through the API and logged, but the registered
PES state assignment does not use it.  It should not be presented as a mass-
aware result.  The registered path always supplies `native_margins`; the
compatibility fallback for `native_margins=None` is outside the registered
configs and must not be used or interpreted as evidence in any result.

## Verified FEPO experiment boundary

The authoritative completed results are in `RESULTS_LEDGER_20260828.md`:

- R18 is the only promoted short-horizon native-relative first-divergence
  reference (`+0.023769` positive cIoU and `+0.090010` utility, complete
  512-row/20k analysis).
- R19, R21-R34, R35, PV, BA, BS and AB are controls or closed branches under
  their stated gates; none establishes a literature-mechanism claim.
- Official RefCOCO/GRefCOCO transfer is reported with its documented
  limitations; it does not turn the holdout result into a universal gain.
- PES normal and shuffled control remain pending while the control plane is
  unavailable.  Their local config/probe checks are implementation evidence,
  not model-quality evidence.

For paper writing, every sentence should therefore be tagged internally as
`paper fact`, `FEPO hypothesis`, or `FEPO result`.  In particular, “inspired
by” is appropriate for the controls above; “implements”, “reproduces”, or
“proves” is not appropriate unless the corresponding mechanism and complete
matched evaluation are present.
