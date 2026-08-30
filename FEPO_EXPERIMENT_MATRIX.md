# FEPO Experiment Matrix

> **Historical snapshot.** The U0/H1/S1 router-and-OPD matrix below belongs to
> the abandoned Idea3/FARO branch. It is retained for provenance only and is
> not an executable plan: current experiments use one SAMTok mask-or-null
> policy, ordinary grouped RL, no OPD teacher, no PixVL trainer, and no
> inference router.

## Current decision matrix (2026-08-29)

| Stage | Paper-derived question | Isolated change | Current decision |
| --- | --- | --- | --- |
| R18 | Does native-relative joint geometry credit improve a SAMTok policy? | cIoU + boundary-IoU credit at first changed code depth, fixed null sentinel | Provisional holdout-selected reference |
| R21-R34 | Do rank, scale, scope, anchor, uncertainty, margin, or null-tail variants add signal? | One registered change per arm, same K=4/5120/10-step contract | All strict screens closed/rejected |
| matched-SFT | Is R18 better than equal-budget continued SFT? | 5,120-row continued SFT control | 512-row/20k overall comparison favors SFT; boundary/thin boundary-IoU tail is non-inferiority risk; retain as control |
| PV-FEPO | Does verified geometry gain persist under a fixed target-preserving photometric view? | Geometric mean of clean/view R18 local credits | Closed by preregistered training support gate: mean joint-positive `0.10625 < 0.20`; no holdout quality claim |
| R35 | Is the remaining limitation representation/null calibration rather than RL credit? | Visual-merger/deepstack LoRA, fixed null/margin protection | Submit only after PV and matched-SFT decisions |
| BA-FEPO | Is PV geometric mean too permissive for boundary-sensitive cases? | Minimum clean/view local credit, boundary bottleneck | Implementation-only; submit only after R35 closes |

Every current stage is required to use at least 5,120 training rows and 10
optimizer steps. Evaluation is exactly 512 image-disjoint rows with 20,000
paired bootstrap repetitions and fixed slice/canonical/null gates. New jobs
must use `dna-` names, `ailab-dnacoding`, and every positive tag in
`rjob_tags.txt`; aggregate live GPU use stays within 24 and artifacts stay
under `Faro_ailab`.

This is the decision gate for the next FARO iteration. Every arm must use the
same checkpoint, examples, rollout groups, decoder settings, optimizer updates,
and evaluator. The current hard `atom_conditioned` route is a baseline, not the
proposed method.

## Candidate methods

| Arm | Router | Local credit | Purpose |
| --- | --- | --- | --- |
| U0 | none | shared reward / OPD | matched unified baseline |
| H1 | current hard bucket | current scalar scales | reproduce Idea3 |
| S1 | predicted-only soft evidence | dominant-branch local scales | legacy FEPO diagnostic |
| S1b | predicted-only soft evidence | `soft_local` weighted scales | unified FEPO main test |
| S2 | shuffled S1 weights | same local scopes | removes route-label confound |
| O1 | GT/overlay oracle | same local scopes | upper bound only; never deployable |
| R1 | predicted-only soft evidence | stable group reward without local scopes | tests whether gains are only reward smoothing |

## Pre-registered success gate

S1 is worth scaling only if, at the same checkpoint:

1. it improves at least two registered failure slices over U0 and H1;
2. S2 loses the improvement under the same compute;
3. DLC negative/no-target calibration does not regress;
4. O1 is better than S1, but S1 retains a non-trivial fraction of the oracle
   headroom; otherwise the evidence signal is not deployable enough;
5. R1 cannot explain the full gain without local token scopes.

The primary table must report RefCOCO/+/g cIoU and AP50, relation and geometry
slices, DLC positive and negative scores, no-target error, and one general
capability recovery set. Do not mix the legacy and vLLM judge protocols.

## Evidence contract

The deployment router may consume only rollout-time predictions, logits,
candidate/confuser scores, and image/text features. GT mask area, GT boundary,
GT caption, and source labels belong in O1 diagnostics only. The evidence
vector is:

```text
semantic = missing coverage + unsupported claims + no-target error
relation = 1 - target/confuser margin
geometry = boundary error + area error
```

The route is a soft mixture. A clean sample keeps the ordinary policy loss;
only failed samples receive local correction. The promoted `soft_local` variant
also mixes CE/RL/OPD scales by the full evidence vector around a neutral shared
scale; dominant-bucket scaling is retained as an ablation. A privileged
teacher may scale the correction magnitude but cannot select its sign or route.

## Execution order

1. Run the CPU contract smoke and static checks.
2. Submit one `dna-` 8-GPU, 200-step matched-data smoke containing U0/H1/S1b/S2.
3. If S1 passes the gate, add O1/R1 and the three component ablations.
4. Add the short caption/general-capability restoration stage.
5. Scale SAV-first only after a fixed-checkpoint table is complete.

All cluster submissions use `ailab-dnacoding`, `dnacoding_gpu`, the shared
dnacoding and wuyucheng mounts, and a job name beginning with `dna-`. Each
submitted job must request at least 8 and at most 24 GPUs; the aggregate live
GPU budget is at most 24 and storage is capped at 700G. Do not submit a new
job while an existing job would make either cap ambiguous.
