# TB-GPPO Decision Audit and One-Stage Contract (2026-08-19)

## 1. Decision

The ES-GR-CPPO 20-step result must trigger Candidate C, **Tail-Balanced
Geometry PPO with selective risk**.

The relevant comparison is ES against the matched-data/update continued-SFT
control, not ES against the frozen anchor alone. This control is not
forward-pass matched; a surviving paper candidate still requires a separate
compute-matched control:

| delta (ES minus matched SFT control) | mean | paired 95% CI | decision |
|---|---:|---:|---|
| selective utility | `+0.01307` | `[+0.00368,+0.02394]` | calibration gain is real |
| positive cIoU | `-0.00121` | `[-0.00482,+0.00179]` | no positive geometry gain |
| no-target recall | `+0.02734` | `[+0.00781,+0.05078]` | null behavior improves |

ES versus the frozen 500-step anchor also passes the development screen (the
candidate-only null repairs and utility gain are real), but that comparison
cannot establish geometry improvement over matched SFT. The correct
interpretation is: **effective-support exploration makes group-relative RL
valid and improves abstention/calibration, while its positive geometry signal
is not yet reliable**.

This strictly enters Candidate C because its preregistered entry condition is
“null behavior is preserved, but positive cIoU does not improve.” It does not
enter PASS-PO: PASS-PO is reserved for a positive geometry signal followed by
an unsafe realized Adam update that loses null behavior. Here null recall
improves and positive cIoU moves slightly downward. Post-Adam backtracking
cannot explain or repair the observed lack of positive signal.

Do not run a 100-step ES/TB job now. The 20-step candidate has not met the
positive-geometry screen, and scaling it would convert a calibration result
into an underpowered geometry claim. The next GPU experiment is only the
one-stage TB-GPPO feasibility test below, followed by its required 20-step
matched controls if and only if the one-step contract passes.

## 2. Fixed lineage and no holdout tuning

- Initialization: the frozen `continued_sft_to500` adapter, with the existing
  SHA256-locked provenance.
- Policy: one shared SAMTok mask-or-null policy; no inference router, expert,
  PixVL checkpoint, verifier, self-supervised cycle, or counterfactual rollout.
- Rollout: the already validated grammar-constrained ES-GR-CPPO K=4 sampled
  mask trajectories, two policy epochs, detached behavior log probabilities,
  and first-null-token/mask-start null constraint.
- Data: the image-disjoint training pairs only. Geometry labels below are
  computed before training from positive training masks and stored in a
  manifest. The 512-row holdout is never used to set a quartile, FIFO value,
  boundary width, hard-label mapping, stopping point, or risk budget.
- Every development evaluation uses all 512 paired rows. No 16-, 32-, or
  64-row result can promote this candidate.

## 3. TB-GPPO one-stage implementation contract

### 3.1 Candidate objective

For each sampled positive trajectory, compute cIoU and boundary IoU. Maintain
two separate empirical rank queues and use a frozen-within-group rank score:

```text
r_geom = w_ciou(hard_flag) * rank16(ciou)
       + w_boundary(hard_flag) * rank16(boundary_iou)
```

The weights are fixed before the job and are not swept:

```text
ordinary group:      w_ciou=0.60, w_boundary=0.40
hard-geometry group: w_ciou=0.40, w_boundary=0.60
```

Group-standardize `r_geom` across the K=4 trajectories and use it as the PPO
advantage. No-target rows receive no rollout advantage. They retain canonical
null CE and the first-action margin constraint from ES-GR-CPPO. The candidate
also logs a lower-tail risk statistic over a fixed 32-row training-only null
sentinel buffer:

```text
risk_null = mean(first_action_margin_hinge)
           + 0.5 * ReLU(-0.05
                        - q_0.10(current_margin - anchor_margin))
```

Equivalently, the second term penalizes the 90th percentile of margin
degradation above `0.05`. The risk term is a constraint diagnostic and loss
term with fixed coefficient `0.25`; it is not a deployment decision rule. The
sentinel IDs and anchor margin values are frozen before training. If a future
implementation cannot fit the 32-row sentinel in one forward pass, it must use
deterministic microbatches and aggregate their sufficient statistics; it may
not replace the tail with a holdout estimate.

### 3.2 FIFO16 initialization

The queues are not initialized empty and are not initialized from the 512-row
holdout. Before the first optimizer step:

1. Select 16 positive training IDs by ascending SHA256 of `pair_id`, after
   excluding the null sentinel IDs. This selection is recorded in the
   manifest and is identical for all arms.
2. Run the frozen 500-step anchor greedily on those IDs once. Decode the masks
   and compute each ID's cIoU and boundary IoU against its GT mask.
3. Initialize `Q_ciou` and `Q_boundary` with those 16 values in the sorted ID
   order. Queue capacity is exactly 16; all values are float32 in `[0,1]`.

This anchor-calibration initialization supplies a fixed training-only reward
scale and avoids the arbitrary first-group behavior of an empty ranker. It is
not a privileged rollout target: the anchor values never change the sign of a
candidate advantage.

### 3.3 FIFO16 ranking and update order

For every positive prompt group, snapshot both queues before scoring any of its
K trajectories. For each component `x`, define the midrank against the
snapshot `Q`:

```text
rank16(x; Q) = (count(q < x) + 0.5 * count(q == x)) / 16.
```

Equal float32 values use exact equality. The rank is in `[0,1]`; no min-max
normalization or reward clipping is allowed. All K trajectories in the group
use the same queue snapshot, so their relative advantage cannot depend on
within-group processing order.

Only after all K rewards, ranks, and advantages for the group have been
computed, append the K raw cIoU values and then the K raw boundary-IoU values
to their respective queues in rollout index order `0..K-1`. Evict the oldest
entries until length is 16. The queue is updated once per group, never between
the two PPO epochs, and never from validation or holdout rows. Log queue
contents hashes, pre/post means, and lengths for reproducibility.

The shuffled-label control uses the exact same queue snapshots and update
order. Its only change is the deterministic permutation of `hard_flag`; this
prevents FIFO timing from explaining a candidate-control difference.

## 4. Boundary IoU definition

Boundary IoU is a fixed geometry metric, not a copied PixVL mechanism. For a
binary mask `M` at the SAMTok decoded resolution, use a 3x3 square structuring
element and exactly `width=2` iterations:

```text
B(M) = (dilate(M, width=2) > 0) AND (erode(M, width=2) == 0)
```

For predicted and GT boundaries `Bp`, `Bg`:

```text
boundary_iou = |Bp AND Bg| / |Bp OR Bg|
```

Use the explicit empty convention: both empty gives `1.0`; exactly one empty
gives `0.0`; otherwise use the intersection/union ratio. The same binary mask
resolution, structuring element, padding rule, and width are used for reward,
diagnostics, and all arms. A future implementation must provide a standalone
NumPy/SciPy-compatible helper and unit tests for empty, full, one-pixel, and
one-pixel-shifted masks; it must not import `projects.pixvl_*` routing code.

## 5. Hard-geometry labels

Labels are training-only strata, computed once from each positive GT mask.
Let `A` be foreground pixel count, `H*W` image-mask area, `P` the perimeter
from the same width-2 boundary convention, and `B=|B(M)|`:

```text
area_ratio = A / (H*W)
compactness = 4*pi*A / max(P*P, 1)
boundary_density = B / max(sqrt(A), 1)
```

Compute the 25th/75th percentiles over the complete positive **training** set
with a specified deterministic linear-interpolation quantile implementation.
The fixed flags are:

```text
small       = area_ratio <= Q25(area_ratio)
thin        = compactness <= Q25(compactness)
boundary_hard = boundary_density >= Q75(boundary_density)
hard_geometry = small OR thin OR boundary_hard
```

Zero-area positives are invalid training data and fail preflight. Store all
raw features, thresholds, flags, and source IDs in the manifest. At least 25%
of positive IDs must be hard and at least 25% ordinary; otherwise the one-stage
contract fails rather than silently changing the 1:1 strata ratio.

The candidate consumes `hard_geometry` only as the fixed reward-mixture weight
above. It is not a predicted failure router and is unavailable at inference.

## 6. Same-ID causal controls

All arms use one precomputed schedule of positive pair IDs, one schedule of
null sentinel IDs, the same seed offsets, K=4, queue initialization values,
optimizer updates, and policy epochs. The schedule contains a 1:1 mixture of
hard and ordinary flags and is frozen before any reward is observed.

Run these three arms:

1. **Plain-rank control:** same FIFO16 queues and risk constraint, but fixed
   weights `(0.50, 0.50)` for every group.
2. **TB-GPPO candidate:** true hard flags and fixed `(0.40, 0.60)` versus
   `(0.60, 0.40)` weights.
3. **Shuffled-label control:** exactly the candidate schedule and IDs, but
   hard flags deterministically permuted by `SHA256(pair_id || seed=1907)`.

The null schedule and 32-row sentinel are identical across arms. No arm may
pick its IDs from the 512-row evaluation records. Initialization queue hashes,
trajectory counts, update timing, and total forward/backward updates must
match. Post-update queue contents are logged but are expected to diverge after
the reward treatments produce different policy updates; requiring later hashes
to match would incorrectly forbid the causal effect being tested.

## 7. One-step gate

Use eight positive groups globally (four per GPU), K=4, and eight paired null
rows. The one-step job must pass every condition:

1. Manifest has exactly 16 FIFO initialization IDs, 32 null sentinel IDs,
   fixed quartile thresholds, and the same schedule hash for all arms.
2. Every trajectory is grammar-valid and decodes to finite cIoU and boundary
   IoU in `[0,1]`.
3. All groups use a pre-group FIFO snapshot; no queue update occurs during a
   group or between PPO epochs.
4. Every queue has length 16 before and after its first update, and the update
   evicts exactly K oldest values per component.
5. At least six of eight groups contain two distinct trajectories; at least
   two groups have nonconstant rank rewards with finite nonzero advantages.
6. The positive policy gradient is finite and nonzero before null constraint
   gradients are added. Epoch-1 ratio mean is within `0.002` of one; epoch-2
   median absolute ratio deviation is above `1e-6`; clip fraction is at most
   `0.5`.
7. Boundary empty conventions and hard-label counts pass preflight. The null
   risk statistic is finite and its budget is not chosen from holdout results.

No 512-row evaluation follows a failed one-step gate. A failure closes TB-GPPO
unless it is a demonstrated implementation bug fixed without changing the
registered constants.

## 8. Twenty-step gate and no direct 100-step escalation

If and only if the one-step candidate and both controls pass, run one 20-step
three-arm comparison. Evaluate all 512 rows with greedy temperature-one
decoding. The candidate advances only if:

- utility delta versus plain-rank control is at least `+0.005` with a paired
  CI lower bound above zero;
- mean positive cIoU is noninferior to plain-rank within `0.01` and improves
  both small and thin/boundary slices by at least `+0.01` point estimate;
- the shuffled-label control loses the candidate on both registered slices;
- no-target recall is noninferior within `-0.01`, risk violation rate is below
  `5%`, invalid rate is zero, and positive mask rate is at least `0.99`;
- queue length, update timing, initialization hashes, and schedule hashes
  remain identical across arms; later content hashes remain reproducible per
  arm but need not be equal between arms.

Only a candidate passing this complete 20-step causal gate may receive one
100-step run and the common FEPO paper gate. A utility gain that comes only
from no-target recall, while positive cIoU remains below the matched SFT
control, is a calibration result and does not authorize scaling.

## 9. Why this is not PASS-PO or a reward sweep

PASS-PO repairs an already demonstrated positive policy update that causes a
realized post-Adam null-risk violation. ES demonstrates the opposite pattern:
valid groups, changed PPO ratios, improved null recall, and no positive cIoU
gain. Backtracking would mostly accept the same weak geometry updates and
cannot create candidate diversity or a boundary signal. TB-GPPO instead tests
whether the *positive reward target* is under-resolving hard boundaries.

The FIFO capacity, boundary width, quartiles, weights, queue initialization,
sentinel size, and gates above are fixed before the next job. No 512-row
coefficient, temperature, boundary-width, or threshold sweep is permitted.
