# Representation-Aware FEPO Preregistration (2026-08-19)

## 1. Why the credit branch is closed

The frozen SAMTok continued-SFT anchor reaches `0.770612` positive cIoU and
`0.796875` no-target recall on the registered 512-row holdout.  Fixed-null
effective-support PPO raises no-target recall to `0.816406` and selective
utility to `0.793628`, but positive cIoU remains `0.770850`.  Improvement-only,
preference, gain-weighted preference, greedy-relative, sign-balanced,
boundary-credit, active-set, and tail-balanced variants do not improve this
geometry result.  These failures are sufficiently diverse that another
advantage normalization, reward mixture, support target, or PPO coefficient
sweep is not a new scientific hypothesis.

The shared implementation audit identifies a stronger explanation.  Every
completed standalone arm loads the anchor LoRA only into non-visual language
linears, while both `model.visual` aliases are explicitly frozen.  Consequently
the policy can recalibrate mask-versus-null language actions but cannot adapt
the SAMTok visual-to-language representation that supplies image evidence for
mask-code selection.

## 2. Candidate J: Projector-Plastic FEPO (PP-FEPO)

### Hypothesis

> Pixel-reward optimization improves geometry only when its gradient can
> update the visual-to-language projection that parameterizes SAMTok mask
> tokens; preserving a frozen selective-language anchor should retain the
> already learned null behavior.

This remains one SAMTok mask-or-null policy.  It adds no router, verifier,
counterfactual sample, decoder, external visual feature, PixVL checkpoint, or
self-supervised cycle.

### Registered plastic subspace

The Qwen3-VL SAMTok base has a 24-block visual encoder followed by one final
patch merger and three DeepStack mergers.  Each merger has `linear_fc1` and
`linear_fc2`; these eight linear modules are the only new LoRA targets.  The
patch embed, all 24 visual attention/MLP blocks, norms, position embeddings,
SAMTok codec, and language-model anchor LoRA remain frozen.  Visual LoRA rank
is fixed at 16 with alpha 32, dropout zero, and zero-effect initialization.

The original continued-SFT adapter is loaded as a frozen `anchor` adapter.
The new merger adapter is loaded simultaneously as `visual`; only parameters
whose adapter name is `visual` may receive gradients.  The mechanism gate must
show that the combined model is exactly anchor-equivalent before the first
optimizer step.

### Arms

| arm | initialization | trainable parameters | objective |
|---|---|---|---|
| frozen anchor | continued-SFT anchor | none | evaluation only |
| matched projector SFT | continued-SFT anchor | merger LoRA only | canonical paired SFT |
| fixed-null ES | continued-SFT anchor | existing non-visual LoRA | effective-support PPO + fixed null CE/margin |
| PP-FEPO | continued-SFT anchor | merger LoRA only | same fixed-null effective-support PPO |

The first comparison isolates whether visual plasticity helps at all; the
second isolates whether verifiable pixel RL uses that plasticity better than
matched supervised learning.  Update count alone is not an equal-compute claim,
so rollout/model-forward counts are reported separately.

## 3. Sequential gates

### Gate J0: one-step mechanism and memory test

Run PP-FEPO for one outer step on two dnacoding GPUs.  It must satisfy all of:

- exactly eight registered merger targets and no visual-block target;
- the anchor and visual adapters are simultaneously active;
- zero trainable anchor parameters and nonzero trainable visual parameters;
- pre-update logits agree with the frozen anchor within `1e-5` max absolute
  error on the registered positive/negative pair;
- at least one finite nonzero visual-LoRA gradient on each rank;
- no trainable base, codec, full-ViT, or language-anchor parameter;
- the existing effective-support, reward-diversity, epoch-two ratio, clipping,
  and grammar gates pass;
- peak memory fits two H200 GPUs without reducing the registered batch.

Failure closes this implementation.  It authorizes a migration bug fix, not a
larger rank, full-ViT unfreezing, or reward sweep.

### Gate J1: 20-step matched screen

Only after J0 passes, run one 20-outer-step PP-FEPO arm and one projector-SFT
arm with the same 20 paired batches and 40 optimizer updates as PP-FEPO's two
policy epochs.  Both start from the same anchor and zero-effect merger
initialization.  Evaluate both on all 512 registered rows.  PP-FEPO advances
only if, versus both fixed-null ES and projector SFT:

- positive cIoU mean delta is nonnegative;
- positive cIoU paired 95% CI lower bound is above `-0.005`;
- no-target recall paired CI lower bound is above `-0.01`;
- selective utility mean delta is nonnegative;
- positive mask rate is at least `0.99`, invalid rate is zero, and exact
  canonical response rate does not decrease.

No rank, alpha, target-layer, seed, LR, reward, or stopping-point sweep is
allowed on this holdout.

### Gate J2: frozen 100-step paper test

A J1 survivor receives one 100-step run plus a forward-count-matched SFT
control.  The paper gate requires positive cIoU improvement of at least
`+0.005` or a paired CI lower bound above zero, utility CI lower bound above
zero, no-target noninferiority above `-0.01`, mask rate at least `0.99`, zero
invalid outputs, and no canonical-format regression.  Only then run official
GRefCOCO/RefCOCO and small/thin/boundary/corruption slices.

## 4. Paper connection and claim boundary

Qwen3-VL-Seg motivates treating pixel representation and post-training as
separate axes; EVP motivates testing whether boundary geometry is
representation-limited; SenseNova-Vision motivates staged specialization with
capability retention; Fine-R1 motivates stabilizing an anchor before online
RL; DR2Seg motivates continuous geometry reward but does not justify another
ranking sweep; latent-denoising work motivates corruption evaluation only after
clean geometry improves.  PixVL contributes evaluation/data interfaces only.

The potential contribution is not "visual LoRA" by itself.  A publishable
result would show that **where pixel RL is allowed to write** determines whether
selective mask policies merely recalibrate existence or actually improve
geometry, under a frozen shared SAMTok anchor and matched SFT/RL controls.  If
projector SFT matches PP-FEPO, the representation hypothesis survives but the
RL claim fails.  If neither improves geometry, Candidate J is closed and the
next idea must alter the pixel representation objective rather than PPO credit.
