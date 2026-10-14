<!--
Date Created: 2026-04-13 23:02:44+03:00
Date Last Modified: 2026-05-15
-->

# Thesis Workflow Runners

This repository now has two pipeline runners with different scopes.

## Final thesis and experiment results

- [Final thesis](<Thesis Final Version/ECE_CANGA_THESIS_VFINAL.docx>)
- [Five-seed experiment protocol and commands](docs/thesis_three_seed_experiments.md)
- [Saved predictions and experiment summaries](outputs/thesis_three_seed_v1/)
- [Audited results tables and final figures](outputs/thesis_five_seed_revision_20261012/)

Working drafts, advisor feedback, personal documents, administrative forms, downloaded literature, and editorial revision records are kept locally and excluded from version control. Model checkpoints and token caches are also local; predictions, metrics, and reproducibility records remain versioned.

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
