# Evaluation Protocol

## Scope

This document freezes the main thesis benchmark before any new model implementation.

- Primary task: multi-label emotion classification
- Frozen model set: Logistic Regression, XGBoost, Random Forest, BERT, BiLSTM + Attention
- Main comparison metrics: macro-F1, macro ROC-AUC, macro PR-AUC
- Per-label metrics: secondary diagnostics only
- VAD: secondary only, not part of the main comparison table

## Current Repo Assessment

The current supervised path is:

1. `src/analysis/analyze_gemini_unified.py`
   - joins unified perfume-level features with Gemini labels
   - writes `outputs/analysis_out/gemini_unified_merged.csv`
2. `src/training/train_eval_gemini784.py`
   - trains a Stage 7 baseline on that merged file
   - writes metrics and OOF predictions to `outputs/training/`

Important limitation of the current Stage 7 script for benchmark use:

- it uses label-wise `StratifiedKFold` for emotion labels rather than one shared fold manifest for all models
- it mixes the emotion task and VAD regression in one run
- it keeps `anger` in the target list even though the current 784-row benchmark data has 0 positive examples for that label

For that reason, the current Stage 7 script is useful as a repo baseline, but it is not the official benchmark contract.

## Official Benchmark Dataset

- Canonical input table: `outputs/analysis_out/gemini_unified_merged.csv`
- Sample granularity: perfume-level
- Unit of analysis: one row per unique `perfume_id`
- Current dataset status at freeze time:
  - 784 rows
  - 784 unique `perfume_id`
  - 58 numeric feature columns selected from the existing prefixes:
    - `NRC_`
    - `VAD_`
    - `Hartmann_`
    - `Savani_`
    - `GoEmotions_`
    - `Cardiff_`

The 43-row gold packet in `data/annotation_packets/Labeled_Annotation_Packet_strict_v1.csv` is not part of the main comparison table. It may be used later as a secondary sanity check only.

## Frozen Targets

### Official target columns

The official multi-label target set is:

- `joy`
- `sadness`
- `fear`
- `trust`
- `surprise`
- `anticipation`
- `disgust`

### Excluded target column

- `anger`

Reason:

- in the frozen benchmark dataset, `anger` has 0 positive examples out of 784 rows
- that makes it infeasible as an official benchmark target and makes macro ROC-AUC / macro PR-AUC undefined if it is included

`anger` may remain present in raw data or legacy scripts, but it is excluded from the official benchmark target list and from all main-table macro metrics.

### Secondary targets

VAD dimensions are frozen as secondary only:

- `valence`
- `arousal`
- `dominance`

If reported, VAD metrics must be stored separately and must not appear in the main comparison table.

## Frozen Split Design

### Outer evaluation design

Use one shared outer split design for every model:

- splitter: `KFold`
- `n_splits`: 5
- `shuffle`: `true`
- `random_state`: `42`
- split unit: perfume row / `perfume_id`

### Exact fold-generation rule

To make fold assignment reproducible across reruns and future scripts:

1. Load `outputs/analysis_out/gemini_unified_merged.csv`
2. Sort rows by `perfume_id` ascending
3. Apply `KFold(n_splits=5, shuffle=True, random_state=42)`
4. Save the resulting fold manifest to `outputs/model_benchmark/tables/fold_assignments.csv`
5. Reuse that same manifest for every model

### Rationale

- The repo already has a stable perfume-level merged training table.
- The benchmark must use the same row partitions for all models.
- With only 3 positive `fear` examples, no 5-fold design can guarantee positive support in every test fold anyway.
- A single saved fold manifest is the lowest-rework, most comparable option.

### Inner validation rule for models that need early stopping

Classical models may train directly on each outer-train partition.

Models that require a validation subset for early stopping, such as BERT or BiLSTM + Attention, must derive it deterministically from the outer-train partition:

1. Take the outer-train rows only
2. Sort them by `perfume_id` ascending
3. Apply `KFold(n_splits=10, shuffle=True, random_state=42)`
4. Use the first split's test portion as the inner validation subset
5. Save the manifest to `outputs/model_benchmark/tables/{model_id}_inner_val_folds.csv`

This inner validation subset is for training control only. Official benchmark metrics are always computed from outer-fold OOF predictions.

## Frozen Seed Policy

Use a single fixed seed policy for all official runs:

- data split seed: `42`
- model seed: `42`
- threshold: `0.3`

Rules:

- No seed sweeps for the main table
- No reporting the best seed
- No repeated CV in the main table
- Every stochastic library used by a model must be set to seed `42`
- Every run must log the seed values actually used

For neural models, the run log must also record:

- framework version
- device
- deterministic flags, if any

## Frozen Metric Definitions

### Main table metrics

For every model row, compute these three official metrics from OOF predictions over the 7 official target columns:

- macro-F1
- macro ROC-AUC
- macro PR-AUC

Definitions:

- macro-F1: unweighted mean of per-label binary F1 scores using a fixed probability threshold of `0.3`
- macro ROC-AUC: unweighted mean of per-label ROC-AUC values
- macro PR-AUC: unweighted mean of per-label average precision values

### Threshold tuning diagnostic

The fixed `0.3` threshold remains the official main-table baseline. Threshold tuning is an additional diagnostic experiment, reported in the per-model metrics JSON and per-label CSV, not a replacement for the frozen main-table row.

Threshold tuning is leakage-safe:

- thresholds are selected inside each outer fold using inner-validation predictions from that fold's outer-train rows only
- the held-out outer test fold is never used to choose thresholds
- selected thresholds are then applied to that outer fold's test predictions
- both one global threshold across labels and per-label thresholds are reported
- the grid is configured in `evaluation.threshold_tuning`

Rationale: in imbalanced multi-label classification, a fixed baseline threshold may be suboptimal because label prevalences and score distributions differ across labels. Threshold tuning is used as a calibration step to improve final binary decisions under Macro-F1.

### Secondary diagnostics

These may be reported outside the main table:

- per-label F1
- per-label ROC-AUC
- per-label PR-AUC
- micro-F1
- subset accuracy
- hamming loss
- label prevalence summary
- VAD metrics

Per-label metrics are diagnostic only. They do not replace the three frozen main-table scalars.

## Completion Rule

A model row is complete only if all of the following exist:

- reproducible run
- saved metrics
- comparable plots
- short interpretation note

Operationally, a row is complete only if:

- the run used the frozen dataset, targets, folds, and seeds
- required artifacts were saved under `outputs/model_benchmark/`
- the comparison-table completion flags are all `true`

## Comparable Plot Requirement

Each model must save the same minimum plot set:

1. `figures/{model_id}_per_label_metrics.png`
   - grouped bars for per-label F1, ROC-AUC, PR-AUC over the 7 official labels
2. `figures/{model_id}_roc_pr_curves.png`
   - ROC and PR curves generated from the official OOF probabilities

Ad hoc plots are allowed, but they do not replace the required comparable plots.

## Output Root

Official benchmark artifacts live under:

- `outputs/model_benchmark/metrics/`
- `outputs/model_benchmark/figures/`
- `outputs/model_benchmark/predictions/`
- `outputs/model_benchmark/logs/`
- `outputs/model_benchmark/tables/`

See `docs/artifact_contract.md` for the required file contract.
