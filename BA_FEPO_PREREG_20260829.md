# BA-FEPO preregistration (2026-08-29)

## Question

Does strict boundary agreement across a clean image and a fixed
target-preserving photometric view improve the robustness of R18's
native-relative mask-token policy? The candidate tests a boundary-sensitive
failure mode motivated by Qwen3VL-Seg and DR2Seg; it is not a new router.

## Fixed method

BA-FEPO initializes from the approved SAMTok anchor and trains one
mask-or-null policy with K=4 grouped rollouts on exactly 5,120 training rows
for 10 optimizer steps. The native first-divergence credit, grammar,
effective-support calibration, and 32-row no-target sentinel are unchanged
from R18. For the same sampled code trajectory, clean and photometric
geometry are scored against the training mask. The credit is the minimum of
the two independently normalized R18 local credits. No teacher, OPD target,
counterfactual label, EMA, PixVL trainer, or inference route is used.

## Evaluation and gates

Evaluate exactly 512 paired holdout rows. Compute 20,000 paired bootstrap
repetitions for utility, positive cIoU, boundary IoU, and no-target recall,
plus fixed small/thin/boundary-hard slices and canonical-format validity.
Compare against both the frozen R18 policy and the forward-count-matched
continued-SFT control. Require non-inferiority to both references, positive
cIoU and utility lower bounds, no-target recall lower bound above `-0.01`,
invalid-output rate at most `0.01`, positive-mask rate at least `0.95`, and no
registered slice regression. A failed gate closes the paired-view boundary
line; no coefficient sweep is allowed.

## Queue policy

BA-FEPO is not submitted while PV-FEPO or R35 is unresolved. Any future
submission must use a `dna-` name, namespace `ailab-dnacoding`, every positive
tag in `rjob_tags.txt`, and the existing 24-GPU/700G limits. Its evaluator
uses the 8 -> 6 -> 4 -> 2 -> 1 GPU fallback with 300 seconds at each
non-terminal level.
