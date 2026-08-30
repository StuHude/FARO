# Training geometry coverage audit (2026-08-29)

This is a training-only diagnostic. It does not inspect candidate holdout
metrics and does not authorize a new rjob.

## Registered data checks

- `egfepo_train_5120.jsonl`: 5,120 non-empty rows, balanced 2,560 positive
  and 2,560 no-target rows.
- The target-present geometry registry contains 2,560 pair IDs. The current
  R22 registry marks 640 each as `small`, `thin`, and `boundary_hard`.
- Flag overlap is material: 150 pairs are simultaneously small, thin, and
  boundary-hard; 490 are thin+boundary-hard but not small; 490 are small but
  not thin/boundary-hard; 1,430 are ordinary on all three flags.
- The registry therefore supports a fixed hard/ordinary schedule, but the
  three tail labels cannot be interpreted as independent factors.

## Existence leakage check

Using `tools/audit_existence_leakage.py` against the fixed 512-row selective
holdout:

```json
{
  "train_rows": 5120,
  "test_rows": 512,
  "train_balance": 0.5,
  "test_balance": 0.5,
  "query_nb_accuracy": 0.66015625,
  "parent_path_accuracy": 0.5,
  "in_this_image_rule_accuracy": 0.5
}
```

The modest query-only accuracy and chance-level path/phrase rules do not show
an obvious existence shortcut. This remains a diagnostic, not a capability
claim.

## Consequence for candidate ordering

R35 remains the sole next submission while the control plane is unavailable.
If R35 closes, BA-FEPO must keep the pre-registered paired-view bottleneck
unchanged. Any later BS-FEPO sampling study must report the above overlap and
must not claim that a gain belongs uniquely to thin or boundary-hard objects.
