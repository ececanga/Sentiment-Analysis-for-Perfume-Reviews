Archived scripts that are not part of either active runner live here.

- `legacy_scripts/` contains former `src/` files kept for provenance or occasional manual reuse.
- These files are not invoked by `src/pipeline/run_pipeline.py`.
- These files are not invoked by `src/pipeline/run_thesis_workflow.py`.
- `legacy_scripts/training/train_eval_gold43.py` is the completed Gold43 43-perfume supervised baseline, superseded by the active Gemini784 workflow in `src/training/train_eval_gemini784.py`.
- If a script is promoted back into active use, move it under `src/` and document the workflow in `README.md`.
