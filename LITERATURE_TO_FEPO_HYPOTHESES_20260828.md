# Ten-paper synthesis for the SAMTok-only FEPO study

This note fixes the literature boundary for the current experiment ladder. The
papers are used to form falsifiable hypotheses, not as sources of checkpoints,
teachers, PixVL training code, or inference experts. All experiments below
remain a single SAMTok mask-or-null policy.

| Work | Mechanism relevant to this study | FEPO hypothesis | Required control or exclusion |
| --- | --- | --- | --- |
| Qwen3VL-Seg | Separates target-present, negative, and difficult segmentation cases and reports boundary-sensitive behavior. | A pixel policy should be evaluated on positive, null, and geometry-tail slices rather than one mean cIoU. | 512-row positive/null holdout, boundary-IoU and fixed small/thin/boundary slices; no Qwen checkpoint or decoder is imported. |
| PixVL | Demonstrates verifiable pixel-token interfaces and cyclic/self-supervised routing ideas. | Discrete mask tokens permit direct trajectory scoring, but the training loop should be simpler and independently attributable. | PixVL is restricted to approved evaluator/data interfaces; no PixVL weights, trainer, cycle, verifier, OPD target, or self-supervised loop. |
| EVP | Shows that visual representation and boundary/detail quality can bottleneck segmentation. | If policy credit changes null calibration but not boundary quality, the frozen visual merger is the likely bottleneck. | Keep an interface-plasticity arm separate from policy-credit arms and compare against equal-compute projector SFT. |
| SenseNova-Vision | Uses staged post-adaptation checks to retain broad multimodal capability. | A segmentation gain is incomplete if general or existence behavior regresses after specialization. | Run official RefCOCO/GRefCOCO and capability-retention checks only after holdout promotion; report null and positive metrics separately. |
| Fine-R1 | Uses supervised stabilization followed by relative policy optimization (with TAPO-style triplet augmentation in its full method). | After strong SFT, sibling-relative credit is better posed than absolute reward labels, provided exploration has effective support. | K=4 complete rollouts, detached behavior log-probabilities, nonconstant-group and ratio-change gates; FEPO does not claim to implement TAPO or counterfactual labels. |
| DR2Seg | Uses Look-to-Confirm style refinement and distribution-ranked reward to distinguish segmentation quality. | Joint cIoU and boundary-IoU improvements should define positive geometry credit, with rank-based variants testing robustness to reward scale. | R21 native rank-local and R22 scale-stratified screens; FEPO borrows rank feedback only and does not claim to implement Look-to-Confirm. |
| arXiv:2604.21343 | Studies projected visual-token corruption/noise and latent recovery. | This motivates representation-robustness diagnostics, but it is not a view-consistency or teacher target in FEPO. | View-drop/evidence arm R19 is a negative control; no latent denoising or asymmetric-view distillation is trained. |
| OpenWorldSAM | Addresses open-vocabulary/open-world ambiguity and risk-sensitive segmentation evaluation. | Its ambiguity/risk framing motivates protecting null/capability tails, but the training sentinel is FEPO-specific, not an OpenWorldSAM objective. | R26 changes only fixed training-sentinel tail repair; no OOD threshold is selected on the holdout. |
| V-Zero (arXiv:2606.25319) | Uses answer-label-free clean/negative evidence and teacher-side replay in on-policy optimization. | We borrow only the principle that detached evidence can gate *scope*; PES does not reproduce its contrastive views, teacher distribution, or replay. | R25/PES use student-native entropy/margins only; shuffled/none controls are required and no OPD objective or teacher path is allowed. |
| arXiv:2608.14144 (S2VOPD line) | Uses clean-view/augmented-view asymmetry with an EMA clean teacher; severe information removal can destroy evidence. | View asymmetry motivates a bounded robustness diagnostic, not a training target in the SAMTok-only study. | R19 view-drop is a negative control and failed; PES/FEPO include no EMA teacher, asymmetric-view distillation, or self-supervised cycle. |

## Unified research claim

The common object across the ten papers is not a collection of failure-type
routers. It is a verifiable decision trajectory with two coupled risks:

1. **geometry risk:** a sampled mask must improve both overlap and boundary
   quality relative to the native greedy action before it receives positive
   credit;
2. **abstention risk:** updates must remain inside a fixed training-only
   no-target sentinel budget.

FEPO therefore localizes credit in one policy: the native-relative rank is the
sign/strength of the verified geometry gain, the first changed SAMTok code
depth is the action scope, and the null sentinel is a constraint. R21--R28
are independent tests of this decomposition (rank reference, target scale,
coarse/fine scope, anchor drift, uncertainty, and null tail), not modules to
stack after every result.

## Falsification order

1. Reject any run lacking 5,120 training rows, 10 optimizer steps, grammar
   validity, effective support, or finite PPO diagnostics.
2. Reject any candidate whose complete 512-row paired holdout fails the
   preregistered positive-cIoU, utility, null-risk, canonical-format, or
   invalid-output gates.
3. Require 20,000 paired bootstrap intervals and slice diagnostics before an
   official transfer claim.
4. Use RefCOCO/GRefCOCO and capability retention to test whether a holdout
   improvement transfers; a regression is reported, not tuned away.

This ordering prevents evidence gates, scale strata, or uncertainty terms from
being presented as causal when the underlying geometry credit has not survived
its matched controls.
