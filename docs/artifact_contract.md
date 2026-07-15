<!--
Date Created: 2026-04-13 23:02:44+03:00
Date Last Modified: 2026-07-15
-->

# Artifact Contract

## Contract versions and roots

The repository has two benchmark contracts, two versioned Phase 3 diagnostic supplements, and one Phase 5 supervised feature-analysis supplement:

| Contract | Root | Status |
| --- | --- | --- |
| perfume-only v1 | `outputs/model_benchmark/` and historical branch-specific equivalents | legacy/non-final; not thesis evidence |
| source-grouped v2 | `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/` | accepted final benchmark |
| extended baselines v1 | `outputs/source_grouped_extended_baselines_v1/` | post-acceptance supplemental diagnostics |
| all-metric bootstrap v1 | `outputs/source_grouped_all_metric_bootstrap_v1/` | post-acceptance supplemental uncertainty |
| supervised feature ablation v1 | `outputs/source_grouped_feature_ablation_v1/` | post-acceptance supplemental representation analysis |

The v2 root is isolated from every legacy root. Explicit Phase 2 reruns may
replace model-specific files only inside the configured v2 benchmark root and
must refresh their run manifests, hashes, and code-bound stage receipts. A
file under the legacy or archived `outputs/gemini_browser_og784/model_benchmark/`
tree cannot satisfy a v2 artifact requirement. Direct grouped runners must
reject any `--out-root` other than the configured v2 root. See
`docs/evaluation_protocol_source_grouped_v2.md` for the evaluation rules.

## Source-grouped v2 preflight contract

Before a model starts, the exact SHA-256 hashes of the raw transcript CSV,
browser label CSV, and feature-label CSV must match
`config/evaluation_config_gemini_browser_og784_source_grouped_v1.yaml`.

### Source-audit artifacts

These files are mandatory under
`outputs/gemini_browser_og784_source_grouped_v1/source_audit/`:

- `source_memberships.csv`
  - traces canonical `perfume_id` values to the valid video and normalized
    transcript source keys used by `source_identity_v1`
- `source_components.csv`
  - traces every canonical `perfume_id` to its deterministic
    `source_group_id` and component size
- `source_group_audit.json`
  - records the grouping version, normalization rules, frozen input paths and
    hashes, row and source-key counts, component-size distribution, giant
    component, and validation results

Blank or invalid source keys must be counted but must not create graph edges.
The audit JSON must state whether all 784 perfume IDs were resolved and whether
any source key crosses the final outer folds.

### Shared split artifacts

These files are mandatory under the v2 benchmark `tables/` directory and must
exist before any runner is called:

- `fold_assignments.csv`
  - required columns: `perfume_id`, `fold_id`, `Perfume`, `source_group_id`,
    `source_group_size`
  - may include additional source-count audit columns
  - exactly 784 unique perfume IDs and fold IDs exactly 1 through 5
  - each `source_group_id` occurs in one outer fold only
- `inner_validation_folds.csv`
  - required columns: `outer_fold_id`, `perfume_id`, `Perfume`,
    `source_group_id`, `inner_role`
  - `inner_role` is either `train` or `val`
  - contains outer-training rows only for each `outer_fold_id`
  - a source group cannot cross inner `train` and `val`
- `fold_balance.csv`
  - reports at least the number of perfume rows, number of source groups, and
    positive support for every official label by fold
  - must make zero-support rare-label folds and the giant component visible

The outer and inner manifests are shared inputs, not per-model outputs. Model
runners must validate and read them; they must fail if either is absent rather
than generate an ordinary KFold replacement.

The rebuilt deep JSONL and chunk audit CSV must embed official labels, excluded
labels, VAD values, outer folds, and source groups that exactly match their
frozen tables before either deep runner starts. For v2, `chunk_text` and
`document_text` contain transcript excerpts only. Title, description, URL, and
channel fields are audit metadata and cannot be model input.

### V2 per-model provenance additions

The required per-model filenames remain those defined below, relative to the
v2 benchmark root. In v2, both `{model_id}.json` and
`{model_id}_run.json` additionally require:

- `dataset_sha256`
- `config_path`
- `config_sha256`
- `frozen_input_sha256`
- `split_design` equal to `5-fold source-component grouped CV`
- `split_provenance.type`
- `split_provenance.grouping_version` equal to `source_identity_v1`
- `split_provenance.group_column` equal to `source_group_id`
- `split_provenance.outer_manifest_sha256`
- `split_provenance.inner_manifest_sha256`
- `split_provenance.source_audit_sha256`

The v2 OOF prediction CSV additionally requires `source_group_id`. It must copy
`fold_id` and `source_group_id` from the prebuilt outer manifest without
recalculation.

All five completed run manifests must contain identical hashes for the shared
outer manifest, inner manifest, and source audit. Classical models must record
the frozen feature-label CSV hash as
`frozen_input_sha256.feature_label_csv`. Deep models must record
`frozen_input_sha256.raw_csv`, `frozen_input_sha256.labels_csv`, and the rebuilt
deep-text dataset hash. All five must record the same prospective configuration
hash.

The workflow-level `workflow/command_manifest.json` and per-stage receipts additionally record the
base Git commit and SHA-256 of every split builder, model runner, threshold
utility, deep-data builder, and orchestrator used by the execution. This is the
authoritative code snapshot when Phase 2 work is present as a reviewed working
tree on top of the tagged Phase 1 baseline.

Every final model stage receipt is bound to the exact config and current hashes
of the split builder, data builder, threshold utility, five runners,
diagnostic/export utilities, and orchestrator. A partial invocation may update
the command manifest but cannot make stale stage receipts pass the final gate.

### V2 benchmark-wide diagnostics and export

The following are mandatory in addition to the five model rows:

- `dummy_prevalence_baselines/` containing all-negative, all-positive, and
  complementary-training-fold prevalence OOF diagnostics;
- `tables/inner_fold_balance.csv`, including label support for inner `train`
  and `val` roles;
- `tables/outer_fold_per_label_metrics.csv`, with support and explicit defined
  flags; ROC-AUC is blank/null unless both classes occur, and average precision
  is blank/null when positive support is zero;
- `tables/component_weighted_model_metrics.csv`, giving each source component
  equal total row weight;
- `tables/paired_component_bootstrap_macro_f1.csv`, using paired component
  resamples for model-difference intervals;
- `tables/source_grouped_diagnostics_report.json`, with strict JSON hashes and
  method provenance; and
- `tables/final_model_comparison_for_thesis.csv`, whose thesis-facing retrieval
  metric column is `macro_average_precision` rather than `macro_pr_auc`.

### V2 completion rule

The source-grouped comparison table is candidate final evidence only if:

- all frozen input hashes match;
- all source-audit and shared split artifacts exist;
- outer and inner group-overlap checks are zero;
- fold and rare-label support are reported;
- trivial baseline, component-weighted, and paired component-bootstrap
  diagnostics pass their provenance checks;
- all five prespecified model rows satisfy the per-model contract; and
- no required artifact was copied from a legacy benchmark root.

An undefined rare-label fold metric is stored as `null`/`NA`, with support and
the metric denominator reported. It must not be silently converted to zero.
The artifact field historically named `macro_pr_auc` stores average precision;
tables and prose must call it **macro average precision**.

The gate independently recomputes fixed-threshold OOF and per-label metrics,
requires `pred_<label> == (prob_<label> >= 0.3)`, and numerically checks each
comparison-table row. Inner-validation threshold optimization is a secondary
sensitivity analysis, not probability calibration and not the primary decision
rule.

## Phase 3 supplemental artifact contract

The supplemental roots do not replace, rewrite, or satisfy any accepted v2
workflow receipt. They must bind to the frozen 784-row feature-label table and
outer manifest; the uncertainty supplement must additionally validate all five
accepted OOF files and the shared inner manifest.

`outputs/source_grouped_extended_baselines_v1/` requires:

- `extended_baseline_summary.csv`, with exactly the six rules all-negative,
  all-positive, always-common-labels, train-fold prevalence, uniform random,
  and stratified random;
- `extended_baseline_per_label.csv`;
- `extended_baseline_fold_diagnostics.csv`;
- `extended_baseline_oof_predictions.csv`, containing all 784 perfume IDs and
  the frozen fold and source-group IDs;
- `REPORT.json`; and
- `SHA256SUMS`.

The threshold is fixed at 0.3 and random rules use seed 42. The stratified rule
may estimate probabilities only from the complementary outer training folds.

`outputs/source_grouped_all_metric_bootstrap_v1/` requires:

- `model_metric_bootstrap_intervals.csv`, with 15 rows: five models by three
  headline metrics;
- `paired_model_metric_bootstrap_differences.csv`, with 30 rows: ten pairs by
  three headline metrics;
- `REPORT.json`; and
- `SHA256SUMS`.

Both tables must record 2,000 paired source-component repetitions, seed 42,
359 source components, 95% percentile intervals, and the defined-label count.
The macro-F1 rows must reproduce the accepted component-equal point estimates
and all ten accepted paired macro-F1 differences exactly. Macro average
precision is the primary ranking metric, macro-F1 at threshold 0.3 is the
primary thresholded metric, and macro ROC-AUC is a secondary ranking metric.

## Phase 5 supervised feature-analysis contract

`outputs/source_grouped_feature_ablation_v1/` is supplemental and must never
replace or modify the accepted source-grouped benchmark. Its canonical run is
fixed to 2,000 bootstrap repetitions, 30 held-out permutation repetitions,
seed 42, confidence level 0.95, threshold 0.3, the accepted outer manifest,
and the frozen 58-column numeric feature schema. A non-empty canonical output
root may be replaced only when every existing file is owned by and matches its
current `SHA256SUMS` entry.

The analysis requires exactly 15 balanced Logistic Regression variants: the
full hybrid, NRC only, VAD only, NRC plus VAD, each of the four contextual
families separately, all contextual families, and six leave-one-family-out
variants. Each variant must produce a 784-row OOF CSV with the frozen perfume,
fold, source-component, truth, probability, and thresholded-prediction fields.
Before any OOF file is written, the full 58-feature replay must reproduce the
accepted Logistic Regression truth and predictions and match its probabilities
within `1e-12`.

Required top-level artifacts include:

- `feature_columns_by_variant.csv`;
- `ablation_summary.csv`, `ablation_per_label.csv`, and
  `ablation_fold_diagnostics.csv`;
- `model_metric_bootstrap_intervals.csv` and
  `paired_bootstrap_vs_full.csv`;
- the three `fear_excluded_*ablation*` / bootstrap tables;
- `family_permutation_importance.csv`, its repeat-level table, its per-label
  aggregate and repeat-level tables, and both fear-excluded permutation tables;
- standardized coefficient, feature-stability, family-stability, and
  coefficient-vector-stability tables;
- contextual TF-IDF comparison and paired-bootstrap tables;
- the two generated figures;
- `REPORT.json`; and
- `SHA256SUMS`.

Bootstrap tables must use paired source-component resampling, equal total
weight per sampled component, shared draws across variants, percentile
intervals with linear quantile interpolation, and explicit defined-label
counts. The seven-label analysis remains primary; the six-label fear-excluded
tables are a declared rare-label sensitivity. Family permutation jointly
shuffles all columns in one family across perfume rows within each held-out
outer fold. Its 2.5% and 97.5% repeat quantiles are Monte Carlo ranges rather
than confidence intervals. Coefficients are the standardized foldwise
Logistic Regression coefficients and are associational diagnostics.

`REPORT.json` must record the accepted-input hashes, exact hashes of the Phase
5 runner and imported training utilities, the full-hybrid reproduction gate,
the bootstrap and permutation definitions, output-path safety checks, and the
accepted benchmark checksum-manifest hash before and after the run. The two
accepted-manifest hashes must be identical.

## Legacy v1 root

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
