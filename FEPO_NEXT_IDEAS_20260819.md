# FEPO Next-Idea Preregistration (2026-08-19, Standalone Revision)

## 1. Fixed project boundary

The routed FARO line is closed. The active method family is one shared
SAMTok mask-or-null policy. Every trainable arm uses the original SAMTok base
and a SAMTok-only standalone adapter. PixVL weights, trainers, verifiers,
cycle/self-supervised training, choose-one training, counterfactual rollouts,
and inference routers are forbidden. PixVL evaluation code and already
approved data are the only permitted reuse.

The current frozen initialization is
`outputs/samtok_selective/continued_sft_to500/adapter`. Its contract registers
500 cumulative SFT steps (the final continuation records 200 local steps) and
pins the adapter model SHA256 to
`7b409c9f2bc3cf2da61adb9c86270dcda6a1991082d5dca4bb1c8a593ea4dfed`.
Changing this anchor creates a new experiment and invalidates comparisons.

On the standalone 512-row paired holdout the anchor has:

| policy | selective utility | positive cIoU | no-target recall | positive mask rate | invalid rate |
|---|---:|---:|---:|---:|---:|
| frozen total-500-step SFT | 0.78374 | 0.77061 | 0.79688 | 1.00000 | 0.00000 |

These standalone metrics must not be numerically mixed with the earlier
PixVL-trainer result (`0.84473` utility). Different training implementations
and checkpoints make that an historical diagnostic, not the matched control.

## 2. Closed surrogate and resulting hypothesis

The first AM-CPPO smoke was not policy optimization. It selected greedy mask
codes under the ground-truth answer prefix, used the same forward tensor for
current and detached "old" log probabilities, and consequently logged zero
clip fraction at every step. It was reward-weighted teacher-forced imitation.
The earlier full-sequence null margin was also invalid because it compared a
multi-token null phrase with one mask-start token.

After correcting the constraint to the first null token versus
`<|mt_start|>`, the 20-step surrogate still fails decisively against the frozen
500-step anchor:

| delta, surrogate minus anchor | mean | paired 95% CI |
|---|---:|---:|
| selective utility | -0.02300 | [-0.03691, -0.01024] |
| positive cIoU | -0.01867 | [-0.03319, -0.00563] |
| no-target recall | -0.02734 | [-0.05078, -0.00781] |

There are eight anchor-only correct no-target rows and one surrogate-only
repair. The surrogate is rejected and cannot be a baseline called RL, PPO, or
AM-CPPO. Its artifacts remain only as a failure control.

The revised unified hypothesis is:

> Geometry RL can improve a SAMTok mask policy only when complete mask-code
> trajectories receive group-relative credit and every realized optimizer
> update remains inside a directly measured mask-or-null risk region.

The three candidates below are sequential tests of that hypothesis. They are
not modules to stack after every failure.

## 3. Shared protocol and paper gate

- Initialization and reference are the frozen total-500-step adapter above.
- Training data remain the same 256 image-disjoint positive/no-target pairs.
- The 512-row holdout is evaluation-only; no threshold, multiplier, stopping
  point, or checkpoint is selected from it.
- Each positive rollout group is paired with a no-target training example.
- Development evaluation always uses all 512 rows. A one-step or 20-step run
  is only a feasibility screen and cannot support the headline table.
- Report sampled trajectory diversity, nonconstant reward-group rate,
  importance ratios, clip rate, null CE, first-action margin violations,
  positive cIoU, no-target recall, positive mask rate, invalid rate, selective
  utility, discordant counts, and 20,000-repeat paired bootstrap intervals.
- Match both optimizer updates and expensive model forward passes. An
  equal-step SFT control alone is insufficient when a candidate performs four
  rollouts and two policy epochs.
- A 20-step survivor must have utility mean delta at least zero, positive cIoU
  mean delta at least zero, positive cIoU CI lower bound above `-0.01`,
  no-target-recall CI lower bound above `-0.01`, positive mask rate at least
  `0.99`, and invalid rate zero.
- A paper candidate must then pass a frozen 100-step test: utility delta CI
  lower bound above zero, positive cIoU mean delta at least `+0.005` (or its CI
  lower bound above zero), no-target noninferiority CI lower bound above
  `-0.01`, positive mask rate at least `0.99`, and invalid rate zero.
- Only a 100-step survivor advances to 14,229-row official GRefCOCO and full
  RefCOCO. A survivor must beat equal-compute continued-SFT, not only the
  original SAMTok checkpoint.

## 4. Candidate A: Grammar-Rollout Constrained PPO (GR-CPPO)

### Method hypothesis

The surrogate failed because its reward was not attached to a generated
trajectory. GR-CPPO samples `K=4` complete autoregressive SAMTok masks under a
hard output grammar:

```text
<|mt_start|> -> depth-0 code -> depth-1 code -> <|mt_end|>
```

Each later code is conditioned on the sampled prefix. The two sampled code
tokens receive group-standardized cIoU advantage. Detached behavior-policy log
probabilities are retained for two policy epochs, so the second epoch can have
a real PPO ratio. Forced grammar-boundary tokens have probability one under
the constrained action space and are excluded from the importance ratio.

The paired no-target row receives canonical null CE plus a hinge on the first
null-token versus mask-start margin. It receives no binary rollout advantage,
because earlier negative groups were mostly constant and supplied little
policy signal. This is a shared policy and a training-only constraint, not an
outcome router.

### Minimal experiment

```yaml
anchor: continued_sft_to500 (SHA256 pinned)
outer_steps: 1, then 20 only after validity
gpus: 2
rollouts_per_positive: 4
policy_epochs: 2
temperature: 1.0
positive_reward: plain_cIoU
advantage: group_standardized
clip_epsilon: 0.2
null_ce_weight: 1.0
first_action_margin_weight: 0.25
lr: 5.0e-7
```

The one-step job is a mandatory optimizer validity gate. It must produce at
least one nonconstant K=4 reward group, four grammar-valid trajectories per
positive prompt, detached finite behavior log probabilities, finite
advantages, and at least one nonzero epoch-2 ratio deviation. Failure closes
GR-CPPO; it does not authorize temperature or coefficient sweeps.

If the one-step gate passes, run one 20-step job. In addition to the shared
screen, require at least `20%` nonconstant reward groups and at least `10%`
groups with two or more unique code trajectories. Reject if the median
epoch-2 absolute ratio deviation is below `1e-6` (no effective PPO update) or
if clip fraction exceeds `0.5` (unstable update). Compare to the frozen anchor
and a forward-pass-matched continued-SFT control.

### Paper narrative if it survives

The defensible contribution is **constrained group-relative policy
optimization over a discrete pixel-token action space**: full mask trajectories
receive geometry credit while the competing null action is preserved in the
same policy. Do not claim a new PPO algorithm, output grammar, or mask
tokenizer. If GR-CPPO only matches SFT, the result is a rigorous negative
finding about RL after strong selective SFT.

### Candidate A gate result (2026-08-19)

The registered two-GPU one-step job
`dna-samtok-fepo-gr-cppo-one-step-2g-89375344` failed the validity gate.
All eight generated trajectories (K=4 on each rank) were grammar-valid, and
the epoch-two mean absolute importance-ratio deviation was nonzero
(`9.80e-4`), but every K=4 group collapsed to one trajectory. Consequently:

```text
nonconstant_reward_groups = 0
unique_trajectory_mean = 1.0
reward_std_mean = 0.0
advantage_abs_mean = 0.0
policy_loss = 0.0
```

This is a method failure rather than an infrastructure failure. Plain
temperature-one GR-CPPO is closed and its 20-step job is forbidden. The result
identifies a new bottleneck: after the strong 500-step SFT anchor, the
depth-specific SAMTok code policy is too concentrated to supply group-relative
credit. A new candidate may address this exploration collapse only through a
pre-registered training-only entropy rule; an unconstrained temperature sweep
or selection on the 512-row holdout is not allowed.

### Effective-support follow-up and matched-control result (2026-08-19)

The single registered effective-support rule restored valid policy signal. Its
one-step gate passed, and the frozen 20-step run produced nonconstant rewards
in all 160 rollout groups. On the complete 512-row holdout it improved the
anchor's selective utility by `+0.00988` (paired 95% CI
`[+0.00186,+0.01956]`) and no-target recall by `+0.01953`
(`[+0.00391,+0.03906]`), but positive cIoU changed by only `+0.00024`
(`[-0.00456,+0.00569]`). Only 11 of 256 positive greedy outputs changed; five
improved and four degraded, with the gains and losses nearly cancelling.

The 40-update continued-SFT control is also complete. Effective-support PPO
beats it on utility by `+0.01307` (`[+0.00368,+0.02394]`) and no-target recall
by `+0.02734` (`[+0.00781,+0.05078]`), but loses `-0.00121` positive cIoU
(`[-0.00482,+0.00179]`). It therefore fails the registered requirement to
beat matched SFT without losing positive geometry and must not advance directly
to 100 steps. This control matches optimizer updates, not the substantially
larger rollout-forward cost; a paper-scale compute-matched control remains
required.

Four additional positive outputs drift from the anchor's exact canonical mask
response into JSON/code-fence wrappers, although the permissive evaluator can
still extract valid mask tokens. Future gates must report exact canonical
response rate in addition to parser-valid rate.

### Candidate A2: Active-Set Selective-Risk ES-GR-CPPO

The observed movement is mostly abstention calibration while geometry remains
flat. A falsifiable next hypothesis is that applying canonical null CE on every
policy epoch continues optimizing an already strong negative behavior instead
of treating it as a risk constraint. Candidate A2 retains the frozen
effective-support behavior policy, complete sampled mask trajectories, plain
cIoU reward, PPO ratio, and learning rate. It changes only the null protection:

- fix eight no-target sentinel rows by sorted ID from the training JSONL and
  shard them across the two ranks;
- before any update, measure the frozen anchor's sentinel null CE and minimum
  first-action margin;
- set immutable budgets using registered numerical slack, with no holdout
  access;
- before each optimizer update, activate CE and margin gradients independently
  only when the current sentinel violates the corresponding budget;
- require the post-run sentinel to remain inside both budgets.

The one-step job must pass every existing effective-support/PPO validity gate,
log finite anchor/current/final sentinel risks, and pass the final risk gate.
Only then may the frozen 20-step config run. A 20-step survivor must pass the
shared 512-row screen and have positive cIoU at least as high as both the
anchor and the 40-update SFT control; no-target noninferiority remains required.
If it survives, run a no-null-loss arm with identical rollouts and updates. An
active-set arm that is statistically indistinguishable from that arm does not
support a selective-risk novelty claim. The already completed fixed-null ES
run is the other matched objective control.

### Candidate A2 20-step result (2026-08-19)

Candidate A2 is closed. Its two-GPU 20-step training completed all 160 rollout
groups and 40 optimizer updates, with nonconstant rewards in every group and a
passing final eight-row training-sentinel risk gate. Neither active-set
constraint ever activated, so the run is also a direct diagnostic of removing
the always-on null objective under the registered budgets.

The complete 512-row evaluation produced utility `0.79046`, positive cIoU
`0.76842`, no-target recall `0.81250`, positive mask rate `1.0`, and invalid
rate `0.0`. Paired 20,000-repeat bootstrap comparisons are:

| comparison | utility delta (95% CI) | positive cIoU delta (95% CI) | no-target delta (95% CI) |
|---|---:|---:|---:|
| A2 minus frozen anchor | `+0.00672` (`[-0.00051,+0.01561]`) | `-0.00219` (`[-0.00746,+0.00348]`) | `+0.01563` (`[+0.00391,+0.03125]`) |
| A2 minus matched SFT control | `+0.00990` (`[+0.00118,+0.02046]`) | `-0.00364` (`[-0.00812,-0.00007]`) | `+0.02344` (`[+0.00781,+0.04297]`) |
| A2 minus fixed-null ES | `-0.00317` (`[-0.00800,-0.00010]`) | `-0.00243` (`[-0.00606,+0.00042]`) | `-0.00391` (`[-0.01172,0]`) |

Only `250/256` positive outputs used the exact canonical mask-only response,
the same count as fixed-null ES and below the anchor's `254/256`. A2 fails its
registered requirement to match or exceed both anchor and SFT positive cIoU,
and is significantly worse than fixed-null ES in utility. It does not advance
to 100 steps, and its inactive risk set prevents a claimed active-constraint
effect. The failure narrows the next hypothesis: further null-objective tuning
is not justified; any successor must change positive trajectory credit while
retaining the fixed-null ES risk behavior as a control.

## 5. Candidate B: Post-Adam Safe-Step Policy Optimization (PASS-PO)

### Entry condition and hypothesis

PASS-PO is allowed only if GR-CPPO shows nonconstant rewards and a positive
cIoU mean movement but violates null noninferiority or has unstable epoch-2
ratios. V7 already showed that projecting raw gradients does not constrain the
behavior after Adam. PASS-PO therefore evaluates the *realized optimizer
state*, including momentum, rather than another gradient cosine.

For each outer step, generate the GR-CPPO rollouts once and snapshot model,
optimizer, and scheduler states. Propose the full update, then measure null CE,
mean first-action margin loss, and lower-10%-tail margin loss on a rotating,
training-only sentinel buffer. If a budget is violated, restore all states and
retry the same rollout gradients at RL scales `[0.5, 0.25, 0.0]`; null CE and
margin terms remain unchanged. No new or counterfactual rollout is generated.

### Minimal experiment and rejection rule

```yaml
anchor: continued_sft_to500
accepted_outer_steps: 20
rollout_objective: frozen GR-CPPO
backtracking_rl_scales: [1.0, 0.5, 0.25, 0.0]
sentinel: fixed disjoint training-only null buffer
sentinel_rows_per_test: 8
budgets: anchor null CE + fixed numerical slack,
         anchor mean/tail first-action margin - fixed slack
```

Before a 20-step run, a one-step state-integrity test must prove bitwise model
and optimizer restoration after rejection and must prove that a zero-RL retry
matches the null-only control. Reject PASS-PO if fewer than `30%` of accepted
updates retain nonzero RL scale, more than `25%` require scale zero, or actual
forward cost exceeds `2x` GR-CPPO without a utility benefit. At 20 steps it
must pass the shared screen and beat raw GR-CPPO utility by at least `+0.005`;
otherwise the post-Adam mechanism is unnecessary engineering.

### Paper narrative if it survives

The contribution would be **behavior-space safe policy improvement for
mask-or-null generation**, demonstrating that post-optimizer risk measurement
preserves a discrete abstention boundary better than raw-gradient projection.
The claim is empirical and local to the registered SAMTok setting. Do not call
it a formal safety guarantee or a generally convergent constrained optimizer.

## 6. Candidate C: Tail-Balanced Geometry PPO with Selective Risk

### Entry condition and hypothesis

This candidate is allowed only after GR-CPPO or PASS-PO preserves no-target
behavior but fails to make the positive cIoU gain reliable. Qwen3-VL-Seg and
EVP suggest that small, thin, and boundary-heavy regions contain remaining
geometry headroom. Dr. Seg shows that raw heterogeneous continuous rewards can
be dominated by scale and variance. The hypothesis is that short-queue ranks
and lower-tail risk focus RL on that headroom without changing the SAMTok
architecture.

Precompute training-only geometry labels from GT masks: bottom-quartile area,
bottom-quartile compactness, and top-quartile boundary-to-area ratio. Sample
half ordinary positives and half registered hard-geometry positives. Map cIoU
and boundary IoU separately through FIFO empirical CDFs of capacity 16, freeze
queues while scoring each K=4 group, and average the two ranks. Preserve the
surviving null constraint, augmented with a lower-10%-tail first-action-margin
budget so a good mean cannot hide a few new false-mask decisions.

### Minimal experiment and causal controls

```yaml
anchor: continued_sft_to500
outer_steps: 20
rollouts_per_positive: 4
policy_epochs: 2
positive_mix: {ordinary: 0.5, hard_geometry: 0.5}
reward: mean(FIFO16_rank_cIoU, FIFO16_rank_boundary_IoU)
null_risk: mean_margin plus lower_10_percent_margin
```

Run three matched 20-step arms on the identical positive-ID multiset:

1. plain-cIoU GR-CPPO with the surviving risk constraint;
2. tail-balanced ranked geometry PPO;
3. the ranked candidate with hard-geometry labels shuffled.

Reject unless the candidate passes the shared screen, improves both small and
thin/boundary slices by at least `+0.01`, beats the plain-cIoU control by at
least `+0.005` utility, and beats the shuffled-label control on both registered
slices. Also reject if the FIFO queues have fewer than 16 valid historical
values after warmup or if queue updates occur within a rollout group. At 100
steps, both slice deltas need paired CI lower bounds above zero in addition to
the common paper gate.

### Paper narrative if it survives

The claim would be **tail-balanced pixel-policy improvement under selective
risk**, not a new dense decoder. Distribution ranking must be credited to Dr.
Seg; novelty would lie in coupling ranked geometry trajectories with a
mask-or-null lower-tail constraint and proving the effect with shuffled hard-
slice controls. If the shuffled arm matches it, reduce the result to reward
stabilization and do not claim failure-focused learning.

### ES-GR-CPPO 20-step entry evidence (2026-08-19)

The preregistered ES-GR-CPPO successor passed its training-side feasibility
screen and the complete 512-row evaluation. It improved the frozen total-500
SFT anchor, but its positive geometry did not beat the matched SFT update/data
control, so it is not advanced to the 100-step paper gate.

| comparison | utility delta (95% paired CI) | positive cIoU delta (95% paired CI) | no-target recall delta (95% paired CI) |
|---|---:|---:|---:|
| ES minus frozen anchor | `+0.00988` (`[+0.00186,+0.01956]`) | `+0.00024` (`[-0.00456,+0.00569]`) | `+0.01953` (`[+0.00391,+0.03906]`) |
| ES minus matched SFT control | `+0.01307` (`[+0.00368,+0.02394]`) | `-0.00121` (`[-0.00482,+0.00179]`) | `+0.02734` (`[+0.00781,+0.05078]`) |

The matched SFT control itself was not better than the anchor in selective
utility (`-0.00318`, CI `[-0.00977,+0.00169]`) and slightly lost no-target
recall (`-0.00781`, CI `[-0.01953,0]`). Thus ES's utility gain is real in this
holdout and comes primarily from selective abstention, while positive geometry
movement is indistinguishable from zero and below the matched SFT point
estimate. This is the registered entry condition for Candidate C (null
preserved, positive geometry gain insufficient), not for PASS-PO (positive
geometry signal plus realized null-risk degradation).

ES remains the raw exploration/risk control for the TB-GPPO comparison. Its
adapter and all 512-row records are retained; no ES temperature, support,
reward, or seed sweep is authorized.

### Candidate C one-step result (2026-08-19)

The tail-balanced candidate is closed at its one-step selective-risk gate.
The first rerun exposed a measurement-timing bug: the gate averaged sentinel
risk observed before each optimizer update and never measured the final model.
The implementation was corrected without changing any registered method
constant, and the corrected job
`dna-samtok-fepo-tb-tail-one-step-r3-2g-17871-48c0f` measured risk again after
the final optimizer step.

The geometry/RL side was valid: all eight groups had nonconstant rewards and
multiple trajectories, all 32 rollouts were grammar-valid, four rollouts beat
native greedy, the positive policy gradient was nonzero, the epoch-two median
absolute ratio deviation was `0.03442`, and the effective-support controller
hit its target for every decision. However, the final 32-row training-only
null sentinel failed decisively:

```text
final q10(current margin - anchor margin) = -0.25
registered degradation budget             = -0.05
final violation rate                       = 0.53125
```

The second-epoch tail penalty activated, but did not restore the realized
post-optimizer policy to the registered risk region. This is a method failure,
not an infrastructure failure. Per the frozen decision tree, plain-rank and
shuffled-label arms are not run, no 20-step TB-GPPO job is allowed, and no
coefficient, queue, K, support, temperature, boundary-width, seed, or learning
rate sweep is authorized.

## 6.1 Candidate D: Boundary-Credit ES-GR-CPPO

### Entry evidence and hypothesis

Candidate A2 shows that removing the always-on null objective does not release
positive geometry: it lowers positive cIoU versus both the anchor and matched
SFT, and is significantly worse than fixed-null ES in utility. Candidate C
shows that adding a lower-tail penalty makes the realized null boundary unsafe
after one update. The remaining minimally changed hypothesis is therefore on
positive credit alone:

> Effective-support trajectory PPO may need a boundary-sensitive reward to
> distinguish masks with similar region overlap, while the already successful
> fixed-null ES constraint is retained unchanged.

Boundary-Credit ES-GR-CPPO keeps the frozen anchor, positive/null paired batch,
K=4 exploration, top-8 support, target effective support 4, two PPO epochs,
learning rate, clipping, canonical null CE, and first-action margin term from
fixed-null ES. Its only algorithmic change is the positive reward:

```text
reward = 0.5 * raw_cIoU + 0.5 * raw_boundary_IoU(width=2)
```

There is no FIFO, empirical rank, hard/easy label, inference router, active
set, tail penalty, counterfactual rollout, or PixVL training component. The
equal weights and width are frozen before the one-step run; no reward-weight
or width sweep is allowed.

### Sequential gate

Run exactly one two-GPU, one-step validity job first. It must meet every
effective-support gate used by ES, log finite raw cIoU/boundary IoU, have at
least six multitrajectory groups, at least two nonconstant reward groups, a
nonzero positive-policy gradient, and a changed epoch-two ratio. Failure closes
Candidate D.

Only a passing one-step job permits the fixed 20-step configuration. Its full
512-row gate is stricter than A2: utility must not fall below fixed-null ES,
positive cIoU must be noninferior within `-0.01` and have a nonnegative point
delta versus both fixed-null ES and matched SFT, no-target CI lower bound must
exceed `-0.01`, positive mask rate must be at least `0.99`, and invalid rate
must be zero. A surviving 20-step candidate then requires a matched
boundary-reward SFT control before any 100-step claim.

### Candidate D result (2026-08-19)

Candidate D is closed. The one-step and 20-step training gates both passed;
the 20-step run had `160/160` nonconstant and multitrajectory groups, 130
sampled rollouts above the native-greedy mixed reward, nonzero policy gradients,
and effective-support target fraction `1.0`. Its complete 512-row result was:

```text
selective utility  0.793407
positive cIoU      0.770408
no-target recall   0.816406
positive mask rate 1.000000
invalid rate       0.000000
```

Relative to fixed-null ES, utility changed by `-0.00022` (95% CI
`[-0.00066,0]`), positive cIoU by `-0.00044` (`[-0.00133,0]`), and no-target
recall by exactly zero. Only one of 256 positive greedy outputs changed, and
it degraded from cIoU `0.53080` to `0.41748`; no positive output improved.
Relative to the matched SFT control, positive cIoU remained `-0.00165` lower.

Thus the boundary-sensitive rollout reward is valid but does not transfer its
sampled improvements into the greedy policy. It fails the registered
nonnegative positive-cIoU point-delta requirements and cannot advance to 100
steps. No boundary-weight or width sweep is permitted. Together with fixed-null
ES and A2, the result identifies the next bottleneck as credit transfer from
better sampled trajectories, not exploration, null calibration, or reward
availability.

## 6.2 Candidate E: Improvement-Only ES-PPO

### Hypothesis

Across fixed-null ES and Boundary-Credit ES, sampled trajectories frequently
beat the current native greedy decode, yet the greedy policy barely changes or
changes in the wrong direction. Group-standardized advantages assign negative
credit to below-group-mean trajectories; with a concentrated policy this can
cancel the useful positive updates. Candidate E tests credit transfer directly:

```text
advantage = relu(sampled_cIoU - native_greedy_cIoU - 1e-4)
advantage /= mean(positive advantages in this group)
```

It keeps plain cIoU reward, effective-support exploration, K=4, two PPO epochs,
fixed null CE/margin, learning rate, clipping, and all data schedules from
fixed-null ES. It does not introduce a counterfactual rollout: native greedy is
the policy's ordinary baseline decode already computed for the ES gate. It has
no router, FIFO, boundary mix, hard labels, tail penalty, or PixVL component.

### Gate

One-step must retain the ES validity checks and log a nonzero improvement-only
advantage fraction. A run with zero active advantages is closed. Only a passing
one-step run can receive the exact 20-step configuration and complete 512-row
evaluation. Promotion requires utility and positive cIoU noninferior to fixed
null ES and matched SFT, no-target CI lower bound above `-0.01`, positive mask
rate at least `0.99`, and invalid rate zero. No minimum-improvement threshold,
normalization epsilon, or advantage variant sweep is allowed.

## 7. Sequential decision tree

1. Freeze and hash the total-500-step anchor. This is complete.
2. Mark full-sequence and first-action teacher-forced surrogates rejected. This
   is complete.
3. Run only the one-step GR-CPPO validity gate. This is complete and failed
   because all rollout groups were constant; GR-CPPO is closed.
4. Diagnose the collapsed code posterior on training rows and test one
   pre-registered entropy-controlled rollout rule. It must pass the same
   one-step validity gate before any 20-step run or 512-row evaluation.
5. Use PASS-PO only for the specific pattern "positive geometry signal plus
   realized null-risk failure." Use tail-balanced geometry only for the pattern
   "null preserved plus insufficient positive geometry gain."
6. Close a candidate when its frozen gate fails. Do not tune on the 512-row
   holdout and do not combine B and C before either has independent evidence.
7. Keep aggregate live usage within 24 dnacoding GPUs. Every rjob must use
   `ailab-dnacoding`, a `dna-` name, and positive tags from the repository tag
   file. Evaluation follows the 8 -> 6 -> 4 -> 2 -> 1 queue fallback and always
   uses the complete registered sample set.

## 8. Seven-paper borrowing and claim boundary

| paper | permitted lesson for FEPO | claim or mechanism that FEPO must not appropriate |
|---|---|---|
| Qwen3-VL-Seg | staged adaptation, negative/OOD evaluation, and explicit small/boundary geometry slices | its 17M box-guided decoder, high-resolution fusion, SA1B-CoRS/DeRS construction, or ORS benchmark novelty |
| PixVL | evaluation/data interfaces and the importance of verifiable pixel outputs | cycle verification, mask-to-text/text-to-mask self-training, choose-one cold start, cross-view confusers, learned verifier scores, or PixVL checkpoints |
| EVP | boundary/thin-object evaluation and the possibility that visual representation limits dominate | inverse multi-attention refinement, Stable-Diffusion features, RITA/CLIP alignment, or architecture gains |
| SenseNova-Vision | staged post-training and broad capability-retention evaluation after specialization | native understanding-generation unification, its foundation architecture, or its data/training scale |
| Fine-R1 | stabilize with SFT before group-relative RL; distinguish optimization from representation learning | TAPO, image triplets, intra/inter-class augmentation, CoT training, or counterfactual/nearby-negative rollouts |
| Dr. Seg / DR2Seg line | short-FIFO distribution-ranked continuous rewards and queue-size ablations | Look-to-Confirm or distribution ranking as FEPO inventions; raw-vs-ranked controls remain mandatory |
| Latent Denoising, arXiv:2604.21343 | corruption/OOD robustness testing after a policy survives | saliency corruption, latent reconstruction, teacher patch recovery, contrastive distillation, or denoising training |

The papers support a disciplined combination of strong SFT initialization,
verifiable geometry rewards, staged evaluation, and explicit risk accounting.
They do not jointly imply that FEPO works. Only the registered matched controls
and full-sample confidence intervals can support that conclusion.

## 9. Publishable outcomes

There are only three honest paper outcomes:

1. **GR-CPPO succeeds:** complete pixel-token trajectories improve positive
   geometry while a shared null constraint preserves abstention. This is the
   preferred FEPO result.
2. **PASS-PO or tail balancing succeeds after a diagnosed GR-CPPO failure:**
   the paper centers the single mechanism that repairs that failure, with raw
   GR-CPPO and shuffled/equal-compute controls. It does not present a stack of
   unrelated tricks.
3. **No RL candidate beats the frozen 500-step SFT anchor:** report a rigorous
   negative result showing that stronger supervised selective calibration
   dominates tested pixel RL under matched SAMTok-only constraints. Do not
   manufacture a routing or RL success claim.
