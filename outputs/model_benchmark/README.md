# Model Benchmark Output Contract

This directory is reserved for the official thesis benchmark outputs.

## Subdirectories

- `datasets/`
  - canonical deep-model text inputs derived from the frozen perfume benchmark
- `metrics/`
  - saved scalar metrics and per-label diagnostics
- `figures/`
  - required comparable plots for each model
- `predictions/`
  - official OOF predictions for each model
- `logs/`
  - console logs and run manifests for reproducibility
- `tables/`
  - fold manifest, comparison table, and short interpretation notes

## Row completeness rule

A model row is complete only if it has:

- reproducible run
- saved metrics
- comparable plots
- short interpretation note

Do not place official benchmark outputs anywhere else in `outputs/`.
