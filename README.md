# Thesis Workflow Runners

This repository now has two pipeline runners with different scopes.

## `run_pipeline.py` vs `run_thesis_workflow.py`

- `src/pipeline/run_pipeline.py`
  - Legacy/base runner.
  - Keeps the original behavior for preprocessing, lexicon/VAD, transformers, context terms, and fusion.
  - Scraping remains manual/commented as in the existing project style.

- `src/pipeline/run_thesis_workflow.py`
  - New top-level thesis orchestrator.
  - Includes base stages and newer post-base stages (Gemini labeling, Gemini agreement analysis, Stage 7 supervised training, Stage 8A feature importance).
  - Supports stage listing/selection, dry-run, and continue-on-error.

## Stage Names

Ordered stage registry used by `run_thesis_workflow.py`:

1. `scrape` (optional/manual, excluded by default)
2. `format`
3. `lexicon_vad`
4. `transformers`
5. `context_terms`
6. `top_terms`
7. `reprocess_optional`
8. `fusion_merge`
9. `fusion_aggregate`
10. `gemini_labeling`
11. `gemini_agreement`
12. `supervised_train_eval`
13. `feature_importance`
14. `evaluation_reports`

## Example Commands

List available stages:

```bash
python -m src.pipeline.run_thesis_workflow --list-stages
```

Run default full workflow (excluding `scrape`):

```bash
python -m src.pipeline.run_thesis_workflow
```

Run from a specific stage:

```bash
python -m src.pipeline.run_thesis_workflow --from fusion_aggregate
```

Run selected stages only:

```bash
python -m src.pipeline.run_thesis_workflow --only gemini_agreement supervised_train_eval feature_importance
```

Skip specific stages:

```bash
python -m src.pipeline.run_thesis_workflow --skip scrape reprocess_optional
```

Preview commands without executing:

```bash
python -m src.pipeline.run_thesis_workflow --dry-run
```

Continue despite stage failures:

```bash
python -m src.pipeline.run_thesis_workflow --continue-on-error
```

## Archived Scripts

Legacy scripts that are not part of either runner are kept under `archive/legacy_scripts/`.
Separate non-runner workflows that are still maintained remain under `src/`.
