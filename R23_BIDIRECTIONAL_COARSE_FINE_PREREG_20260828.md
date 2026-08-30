# R23 preregistration: bidirectional coarse/fine native geometry credit

## Hypothesis

R18 assigns a jointly improved mask's entire credit to the first changed
SAMTok code.  This may over-credit an early extent decision when the useful
change is a late boundary refinement.  R23 tests whether crediting both ends
of the changed-code interval improves geometry without adding a router.

For every K=4 sibling rollout, require cIoU and boundary-IoU to both exceed
the native greedy mask by `1e-4`.  Let `d_c` be the first changed code depth and
`d_f` the last changed depth.  The detached scalar advantage uses the existing
native-reference midrank gain multiplied by:

`0.5 * 0.85^d_c + 0.5 * 0.85^(D-1-d_f)`.

The mirrored exponent makes the fine term large when a change occurs near the
end of the code sequence.  Mixed-axis and regressive samples remain neutral;
unchanged trajectories remain neutral.  This is one SAMTok policy and one
grammar-constrained PPO objective, with no PixVL, OPD, counterfactual labels,
visual adapter, inference router, or self-supervised cycle.

## Minimal screen

Use the exact continued-SFT SAMTok anchor, seed 17, `egfepo_train_5120.jsonl`,
5,120 rows, 10 outer steps, two GPUs, four rollouts per prompt, two policy
epochs, effective-support calibration, and the unified 32-row no-target
sentinel.  The only changed field relative to R18 is
`pareto_credit_mode=bidirectional_coarse_fine_native_geometry` with fixed
`coarse_depth_weight=0.5`, `fine_depth_weight=0.5`, and `depth_local_decay=0.85`.
All jobs must use `dna-` names, `ailab-dnacoding`, and every positive tag in
`rjob_tags.txt`; this preregistration does not authorize submission.

Before training, static/config checks must verify stage, rows, world size,
rollout count, policy epochs, anchor identity, and weight values.  Training
must pass all R18 grammar, effective-support, reward-diversity, epoch-two PPO
ratio, LoRA, and unified-sentinel gates.

## Evaluation and gates

Evaluate the complete 512 paired holdout with 20,000 paired bootstrap
resamples against R18.  Report positive cIoU, boundary IoU, selective utility,
no-target recall, invalid-output rate, positive-mask rate, and canonical rate.

Promote only if positive cIoU improves over R18 by at least `+0.005` with a
95% paired CI lower bound above zero, utility CI lower bound is nonnegative,
no-target recall CI lower bound is above `-0.01`, invalid rate is zero, and
positive-mask rate is at least `0.99`.  In addition, at least one registered
small/thin/boundary slice must improve by `>=0.01` with no slice drop larger
than `0.01`.  Otherwise close R23 without weight or decay sweeps.  An effect
that matches a first-depth control is reported as evidence against the
coarse/fine attribution mechanism, not as a new method claim.

## Literature boundary

Fine-R1 motivates grouped sibling-relative policy optimization after a stable
SFT anchor; DR2Seg motivates continuous geometry ranking; Qwen3VL-Seg,
OpenWorldSAM, and EVP motivate explicit boundary/small-object slices; and
SenseNova-Vision plus the latent-denoising work motivate post-adaptation
capability/robustness checks.  R23 borrows none of their architectures,
external features, data recipes, or denoising/OPD objectives.
