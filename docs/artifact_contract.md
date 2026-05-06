# Artifact Contract

## Root

All official benchmark outputs must be written under:

`outputs/model_benchmark/`

## Benchmark-wide artifacts

These files are shared by the benchmark, not tied to one model:

- `outputs/model_benchmark/tables/fold_assignments.csv`
  - required before model comparison starts
  - columns: `perfume_id`, `fold_id`
- `outputs/model_benchmark/tables/comparison_table.csv`
  - the living main-table export
- `outputs/model_benchmark/datasets/perfume_text_dataset_v1.jsonl`
  - canonical perfume-level deep-text input for future `bert` and `bilstm_attention` runs
  - one JSON object per `perfume_id`
- `outputs/model_benchmark/datasets/perfume_text_dataset_v1_chunks.csv`
  - flat chunk-level audit view of the same canonical deep-text input

## Required per-model artifacts

For each `model_id`, the following files are required.

### Metrics

- `outputs/model_benchmark/metrics/{model_id}.json`
- `outputs/model_benchmark/metrics/{model_id}_per_label.csv`

`{model_id}.json` must include at least:

- `model_id`
- `run_id`
- `dataset_path`
- `n_rows`
- `feature_columns`
- `target_columns`
- `excluded_target_columns`
- `split_manifest_path`
- `data_seed`
- `model_seed`
- `threshold`
- `baseline_threshold`
- `macro_f1`
- `macro_roc_auc`
- `macro_pr_auc`
- `secondary_metrics`
- `threshold_tuning`
- `artifact_paths`

`{model_id}_per_label.csv` must include at least:

- `label`
- `support_pos`
- `support_neg`
- `f1`
- `roc_auc`
- `pr_auc`

When threshold tuning is enabled, `{model_id}_per_label.csv` should also include diagnostic columns for baseline and tuned F1 values, including:

- `f1_baseline`
- `f1_tuned_global`
- `f1_tuned_per_label`

### Predictions

- `outputs/model_benchmark/predictions/{model_id}_oof.csv`

Required columns:

- `perfume_id`
- `fold_id`
- `true_<label>` for each official target label
- `prob_<label>` for each official target label
- `pred_<label>` for each official target label

Optional columns:

- `Perfume`
- `pred_tuned_global_<label>`
- `pred_tuned_per_label_<label>`
- model-specific raw scores

### Figures

- `outputs/model_benchmark/figures/{model_id}_per_label_metrics.png`
- `outputs/model_benchmark/figures/{model_id}_roc_pr_curves.png`

These filenames are mandatory so plots remain directly comparable across models.

### Logs and reproducibility

- `outputs/model_benchmark/logs/{model_id}.log`
- `outputs/model_benchmark/logs/{model_id}_run.json`

`{model_id}_run.json` must include at least:

- `run_id`
- `timestamp_utc`
- `git_commit`
- `python_version`
- `package_versions`
- `dataset_path`
- `split_manifest_path`
- `data_seed`
- `model_seed`
- `command`
- `device`

### Interpretation note

- `outputs/model_benchmark/tables/{model_id}_note.md`

Keep it short. Minimum content:

- 1 to 3 sentences
- what worked
- what failed or looked unstable
- whether the model is benchmark-ready for comparison

## Optional artifacts

Optional artifacts do not determine row completeness:

- saved fold models
- saved checkpoints
- VAD outputs
- calibration studies
- threshold sweeps and threshold-tuning diagnostics
- exploratory plots

## Row completeness rule

A main-table row is complete only if these are all true:

- reproducible run artifact present
- metrics artifacts present
- required plots present
- short interpretation note present

If any of those are missing, the row remains incomplete and must not be treated as part of the final benchmark table.
