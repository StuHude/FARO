#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab}
FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
MODEL=${MODEL:-$ROOT/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co}
JOB_NAME=${JOB_NAME:-dna-fepo-eval-sharded-r1}
ADAPTER=${ADAPTER:?ADAPTER is required}
ANCHOR_ADAPTER=${ANCHOR_ADAPTER:-}
OUTPUT=${OUTPUT:?OUTPUT is required}
OUTPUT=$(realpath -m "$OUTPUT")
case "$OUTPUT" in "$FARO_ROOT"/*) ;; *) echo "OUTPUT must be under FARO_ROOT: $FARO_ROOT" >&2; exit 2 ;; esac
[[ "${MODEL,,}" == *samtok* ]] || { echo "MODEL must be the original SAMTok checkpoint" >&2; exit 2; }
if [[ -n "$ANCHOR_ADAPTER" ]]; then
  [[ -f "$ANCHOR_ADAPTER/adapter_config.json" && -f "$ANCHOR_ADAPTER/adapter_model.safetensors" ]] || {
    echo "ANCHOR_ADAPTER is incomplete: $ANCHOR_ADAPTER" >&2; exit 2;
  }
  [[ -f "$ADAPTER/adapter_config.json" && -f "$ADAPTER/adapter_model.safetensors" ]] || {
    echo "Visual ADAPTER is incomplete: $ADAPTER" >&2; exit 2;
  }
fi
if [[ -f "$ADAPTER/adapter_config.json" ]]; then
  adapter_base=$(sed -n 's/.*"base_model_name_or_path": "\([^"]*\)".*/\1/p' "$ADAPTER/adapter_config.json" | head -1)
  [[ "${adapter_base,,}" == *samtok* ]] || { echo "ADAPTER is not SAMTok-derived: $adapter_base" >&2; exit 2; }
fi
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/pixvl_idea3/configs/fepo_eval_deterministic.py}
REFSEG_SCHEMA=${REFSEG_SCHEMA:-$ROOT/data/pixvl_idea3/schemas_eval128/refseg_train_routed.jsonl}
GEOMETRY_SCHEMA=${GEOMETRY_SCHEMA:-$REFSEG_SCHEMA}
MASKCAP_SCHEMA=${MASKCAP_SCHEMA:-$ROOT/data/pixvl_idea3/schemas_eval128/maskcap_train_routed.jsonl}
EXISTENCE_SCHEMA=${EXISTENCE_SCHEMA:-$FARO_ROOT/data/fepo_existence/refcoco_gres_image_disjoint_holdout_256.jsonl}
GEOMETRY_REGISTRY=${GEOMETRY_REGISTRY:-$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_seed17_10step_2gpu/tail_geometry_registry.json}
GPU_COUNT=${GPU_COUNT:-8}
DEFAULT_TAGS_FILE=$FARO_ROOT/rjob_tags.txt
TAGS_FILE=${TAGS_FILE:-$DEFAULT_TAGS_FILE}
[[ -f "$TAGS_FILE" ]] || { echo "Missing positive-tag file: $TAGS_FILE" >&2; exit 2; }
ALL_POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$ALL_POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }
if (( GPU_COUNT < 1 || GPU_COUNT > 24 )); then echo "GPU_COUNT must be between 1 and 24" >&2; exit 2; fi
case "$JOB_NAME" in dna-*) ;; *) echo "JOB_NAME must start with dna-" >&2; exit 2 ;; esac
POSITIVE_TAGS="$ALL_POSITIVE_TAGS"
[[ -n "$POSITIVE_TAGS" ]] || { echo "POSITIVE_TAGS must be non-empty" >&2; exit 2; }
SCHEMA_ARGS=""
if [[ -n "$ANCHOR_ADAPTER" ]]; then SCHEMA_ARGS+=" --anchor-adapter-path '$ANCHOR_ADAPTER'"; fi
if [[ "${REFSEG_SCHEMA,,}" != none ]]; then SCHEMA_ARGS+=" --refseg-overall-schema '$REFSEG_SCHEMA'"; fi
if [[ "${GEOMETRY_SCHEMA,,}" != none ]]; then SCHEMA_ARGS+=" --geometry-schema '$GEOMETRY_SCHEMA'"; fi
if [[ "${MASKCAP_SCHEMA,,}" != none ]]; then SCHEMA_ARGS+=" --semantic-schema '$MASKCAP_SCHEMA'"; fi
if [[ "${EXISTENCE_SCHEMA,,}" != none ]]; then SCHEMA_ARGS+=" --existence-schema '$EXISTENCE_SCHEMA'"; fi
[[ -n "$SCHEMA_ARGS" ]] || { echo "At least one evaluation schema is required" >&2; exit 2; }

rjob submit --name="$JOB_NAME" --namespace=ailab-dnacoding --cpu="$((GPU_COUNT * 10))" --gpu="$GPU_COUNT" --memory="$((GPU_COUNT * 80000))" --positive-tags="$POSITIVE_TAGS" \
  --charged-group=dnacoding_gpu --private-machine=group \
  --mount=gpfs://gpfs1/dnacoding:/mnt/shared-storage-user/dnacoding \
  --mount=gpfs://gpfs1/wuyucheng:/mnt/shared-storage-user/wuyucheng \
  --image=registry.h.pjlab.org.cn/ailab-dnacoding/wuyucheng:test1 \
  --custom-resources=brainpp.cn/fuse=1 --enable-sshd -- bash -lc "
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
cd /opt; mkdir -p /opt/vlm
if [ -f /opt/vlm_env.tar.gz ]; then tar -xzf /opt/vlm_env.tar.gz -C /opt/vlm; rm -f /opt/vlm_env.tar.gz; elif [ -f vlm_env.tar.gz ]; then tar -xzf vlm_env.tar.gz -C /opt/vlm; rm -f vlm_env.tar.gz; fi
/opt/vlm/bin/python /opt/vlm/bin/conda-unpack || true
export PATH=/opt/vlm/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/compat/bin:\$PATH
export LD_LIBRARY_PATH=/opt/vlm/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
export FARO_PROJECT_ROOT='$ROOT'; export FARO_WORKSPACE_ROOT='$FARO_ROOT'; export FARO_SAMTOK_MODEL='$MODEL'
export SAMTOK_SA2VA_ROOT='$ROOT/third_party/Sa2VA'
export PYTHONPATH='$ROOT/.runtime_deps/huggingface_hub_1_21:$ROOT/third_party/transformers/src:$ROOT/third_party/Sa2VA:$ROOT:$FARO_ROOT/Sa2VA:$FARO_ROOT':\${PYTHONPATH:-}
OUT='$OUTPUT'; mkdir -p \"\$(dirname \"\$OUT\")\"; rm -f \"\$OUT\"; TMP=\"\${OUT}.shards\"; rm -rf \"\$TMP\"; mkdir -p \"\$TMP\"
pids=()
for i in \$(seq 0 $((GPU_COUNT - 1))); do
  CUDA_VISIBLE_DEVICES=\$i /opt/vlm/bin/python -m projects.pixvl_idea3.eval.eval_mvp_bundle --config '$CONFIG' --adapter-path '$ADAPTER' $SCHEMA_ARGS --geometry-registry '$GEOMETRY_REGISTRY' --task-id \$i --num-tasks '$GPU_COUNT' --output \"\$TMP/part_\$i.json\" > \"\$TMP/log_\$i.txt\" 2>&1 &
  pids+=(\$!)
done
failed=0
for pid in \"\${pids[@]}\"; do wait \"\$pid\" || failed=1; done
(( failed == 0 )) || { echo \"one or more eval shards failed\" >&2; exit 1; }
[[ \$(find \"\$TMP\" -maxdepth 1 -name 'part_*.json' | wc -l) -eq '$GPU_COUNT' ]] || { echo \"missing eval shard outputs\" >&2; exit 1; }
/opt/vlm/bin/python - \"\$TMP\" \"\$OUT\" <<'PY'
import json, sys
from pathlib import Path
parts = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob('part_*.json'))]
merged = {}
for section in ('geometry', 'semantic', 'refseg_overall', 'relation', 'dlc', 'existence'):
    rows = []
    for payload in parts:
        rows.extend(payload.get(section, {}).get('records', []))
    if not rows:
        continue
    if section == 'existence':
        pos = [r for r in rows if r.get('truth_exists')]
        neg = [r for r in rows if not r.get('truth_exists')]
        pos_recall = sum(bool(r.get('pred_exists')) for r in pos) / max(len(pos), 1)
        neg_recall = sum(not bool(r.get('pred_exists')) for r in neg) / max(len(neg), 1)
        merged[section] = {'num_samples': len(rows), 'records': rows, 'positive_recall': pos_recall, 'no_target_recall': neg_recall, 'balanced_accuracy': 0.5 * (pos_recall + neg_recall)}
    else:
        metric = 'reward' if section == 'semantic' else 'ciou'
        merged[section] = {'num_samples': len(rows), 'records': rows, ('mean_reward' if metric == 'reward' else 'mean_ciou'): sum(float(r[metric]) for r in rows) / len(rows)}
        if section in {'geometry', 'refseg_overall'} and any('truth_exists' in r for r in rows):
            pos = [r for r in rows if r.get('truth_exists')]
            neg = [r for r in rows if not r.get('truth_exists')]
            merged[section].update({
                'positive_mean_ciou': sum(float(r['ciou']) for r in pos) / max(len(pos), 1),
                'positive_mask_rate': sum(bool(r.get('pred_exists')) for r in pos) / max(len(pos), 1),
                'no_target_explicit_recall': sum(bool(r.get('explicit_null')) for r in neg) / max(len(neg), 1),
                'invalid_output_rate': sum(not (bool(r.get('valid_mask_tokens')) or bool(r.get('explicit_null'))) for r in rows) / len(rows),
                'canonical_response_rate': sum(bool(r.get('canonical_response')) for r in rows) / len(rows),
                'positive_canonical_response_rate': sum(bool(r.get('canonical_response')) for r in pos) / max(len(pos), 1),
                'negative_canonical_response_rate': sum(bool(r.get('canonical_response')) for r in neg) / max(len(neg), 1),
            })
            # Preserve fixed, training-independent geometry diagnostics for the
            # complete holdout. These are descriptive slices, never used to
            # tune a checkpoint or promotion threshold.
            slice_rows = {}
            for row in rows:
                metadata = row.get('slice_metadata') or {}
                names = ['all']
                names.extend(name for name in ('small', 'thin', 'boundary_hard') if metadata.get(name))
                if metadata.get('area_stratum') in {'small', 'medium', 'large'}:
                    names.append('area_' + str(metadata['area_stratum']))
                for name in names:
                    slice_rows.setdefault(name, []).append(row)
            merged[section]['slices'] = {}
            for name, slice_data in sorted(slice_rows.items()):
                positives_slice = [r for r in slice_data if r.get('truth_exists')]
                boundary_values = [float(r['boundary_iou']) for r in positives_slice if r.get('boundary_iou') is not None]
                merged[section]['slices'][name] = {
                    'num_samples': len(slice_data),
                    'positive_num_samples': len(positives_slice),
                    'mean_ciou': sum(float(r['ciou']) for r in slice_data) / max(len(slice_data), 1),
                    'positive_mean_ciou': sum(float(r['ciou']) for r in positives_slice) / max(len(positives_slice), 1),
                    'mean_boundary_iou': sum(boundary_values) / max(len(boundary_values), 1),
                }
Path(sys.argv[2]).write_text(json.dumps(merged, indent=2), encoding='utf-8')
PY
"
