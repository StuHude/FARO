# Adapter cleanup (2026-08-29)

The failed screen candidates retain their metrics, provenance, logs, and
holdout evidence, but their duplicated LoRA weight tensors are removed. The
following runs keep complete adapters because they are still needed for an
open decision or reproducibility:

- `continued_sft*` controls and anchors;
- `fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_seed17_10step_2gpu` (R18);
- `fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_seed17_100step_2gpu` (R18 confirmation);
- `fepo_tb_gppo_plain_rank_unified_paired_view_10step_2gpu` (PV-FEPO training
  diagnostics and provenance; its preregistered support gate is closed, so it
  has no holdout or paper claim).

No files under `PixVL_ailab` are modified. The approved cleanup scope is to
remove only `adapter/adapter_model.safetensors` from closed candidates while
leaving their metadata for the results ledger; this deletion has not been
performed from the login node.

The remaining PV adapter is retained temporarily for reproducibility and
because irreversible deletion is not performed from the login node. Storage
is currently about 38G, well below the 700G cap; it can be removed after the
R35/BA decision is complete without affecting any registered evaluation.
