# Storage cleanup plan (2026-08-29)

Current `Faro_ailab` usage is approximately 37G; the 700G ceiling is not
currently at risk. About 30.5G is duplicate SAMTok adapter weights from
closed screens. Their metrics, provenance manifests, logs, and completed
holdout/bootstrap outputs are the auditable experiment record.

## Retain

- `continued_sft_to500` (frozen SAMTok initialization anchor)
- `continued_sft_r18_matched_200` (matched-budget control awaiting holdout)
- `fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_10step_2gpu`
  (R16 reference)
- `fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_seed17_10step_2gpu`
  (R18 reference)
- `fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_seed17_100step_2gpu`
  (R18 long confirmation)
- `fepo_tb_gppo_plain_rank_unified_paired_view_10step_2gpu` (PV-FEPO pending)
- baseline/control adapters needed for any already registered transfer audit

## Candidate for later deletion

For every closed candidate not listed above, remove only
`adapter/adapter_model.safetensors` (and any `adapter_model.bin`) after the
matched-SFT and PV-FEPO decisions are finalized. Keep `adapter_config.json`,
`metrics.json`, `provenance_manifest.json`, and all evaluation artifacts. This
would reclaim roughly 25-30G without deleting scientific evidence.

No deletion is performed automatically in this continuation because broad
irreversible removal requires explicit confirmation of the keep list.
