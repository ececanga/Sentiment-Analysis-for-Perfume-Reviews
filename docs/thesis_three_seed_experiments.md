# Five-seed thesis experiments

The runner now uses training seeds **42, 123, 2026, 7, 99**, with data/split seed **42**. Seeds 7 and 99 are additional, prespecified random seeds; their numeric values have no special statistical meaning. All **117 jobs** for the original three seeds were completed and validated before this extension. Only **78 additional jobs** are needed. The accepted model protocols and historical result trees remain unchanged.

The module, configuration, and output directory retain their `three_seed` names so existing commands and saved runs continue to work. The configured seed list controls inventory and summary completeness. No new full-data training is launched by the extension's tests or dry run.

Run commands from the project directory with `.venv/bin/python`. No additional packages are required.

```sh
# Inspect all jobs and validate inputs, split assignments, and historical reuse.
.venv/bin/python -m src.training.run_thesis_three_seed --dry-run

# Run the missing seeds, preserving all validated completed experiments.
.venv/bin/python -m src.training.run_thesis_three_seed --execute --resume --reuse-seed42

# Request a graceful pause from another terminal (Ctrl-C also requests a pause).
.venv/bin/python -m src.training.run_thesis_three_seed --pause

# Read the latest persistent status; this never trains anything.
.venv/bin/python -m src.training.run_thesis_three_seed --status

# Resume: validate completed jobs and continue incomplete matching checkpoints.
.venv/bin/python -m src.training.run_thesis_three_seed --execute --resume --reuse-seed42

# Recompute summaries from completed, validated per-seed OOF predictions.
.venv/bin/python -m src.training.run_thesis_three_seed --summarize
```

`--reuse-seed42` permits importing historical results only for missing seed-42 jobs. Omitting it does **not** retrain any already completed job under `--resume`. Reuse is never inferred solely from a filename or seed label. A reuse import records the source prediction and evidence hashes in `complete.json` and `training_audit.json`. Historical files are read, never moved or overwritten. `--resume` skips only complete runs whose configuration, code, input hashes, software, prediction completeness, targets, and recomputed metrics validate. An incompatible run raises an error. There is no overwrite switch.

Selection examples (the same selectors work with `--execute`, `--resume`, or `--summarize`):

```sh
.venv/bin/python -m src.training.run_thesis_three_seed --dry-run --experiments main --models bert bilstm_attention --seeds 7 99
.venv/bin/python -m src.training.run_thesis_three_seed --dry-run --models random_forest --conditions youtube_plus_weighted_synthetic_784x3
.venv/bin/python -m src.training.run_thesis_three_seed --dry-run --experiments structured_ablation --models logistic_regression --variants nrc_only transformer_only
```

Use the same selectors when resuming a subset. Running the unfiltered resume command continues the entire inventory. A summary for a subset still flags fewer than the **five configured seeds** as incomplete, listing the missing seeds explicitly. Each invocation refreshes the inventory/audit/summary/status for its selection; the unfiltered dry-run command restores the full inventory and summaries without training. The inventory is also refreshed after each completed job.

## Inventory and reuse audit

The default inventory contains **195 jobs**:

| Experiment | Models/variants | Conditions | Seeds | Jobs |
| --- | --- | --- | --- | --- |
| Main | LR, XGBoost, Random Forest, BERT, BiLSTM+Attention | YouTube-only and weighted YouTube+synthetic | 5 | 50 |
| Structured LR | Existing 15 feature-family variants | YouTube-only | 5 | 75 |
| Structured XGBoost | Existing 7 feature combinations | YouTube-only | 5 | 35 |
| Structured Random Forest | Existing 7 feature combinations | YouTube-only | 5 | 35 |

The original 117 completed jobs comprise **39 audited historical seed-42 imports** and **78 newly trained jobs at seeds 123 and 2026**. The extension adds **78 jobs at seeds 7 and 99**. On a completely fresh output root, there would instead be 156 training jobs plus 39 eligible imports. Full-hybrid ablation controls remain explicit inventory entries.

Read `outputs/thesis_three_seed_v1/seed42_reuse_audit.md` for individual decisions and `seed42_reuse_audit.json` for source paths, hashes, recomputed metrics, software versions, and reasons. `inventory.csv` and `inventory.json` list every job, its exact command, configuration fingerprint, and output directory. The audit is rerun before execution; changed evidence can invalidate a previously reusable result.

Outputs use:

```text
outputs/thesis_three_seed_v1/<experiment>/<model>/<condition>/<variant>/data42_model<seed>/
```

Each completed job has `contract.json`, `oof.csv`, `metrics.json`, `training_audit.json`, and `complete.json`. Checkpoints stay within that job directory. The main deep-model variant is named `standard`; classical full-feature runs use `full_hybrid`.

## Compatibility and preserved three-seed evidence

Before editing the runner, all 117 completed jobs passed their existing contract, artifact-hash, row/label/split, and recomputed-metric checks. `outputs/thesis_three_seed_v1/five_seed_extension/before_extension/` preserves the original three-seed summaries, metadata, configuration, runner source, validated contracts, and hashes of each job's five primary artifacts. Original per-job outputs remain in place and unchanged.

The compatibility registry adds exact old/new contract pairs for these **completed** jobs only. The old validated contracts and the new contracts differ only in the orchestrator's code hash; model training functions, inputs, settings, software, splits, and checkpoint identities are unchanged. Earlier scoped repair approvals remain recorded. No partially trained job is approved for migration across this extension. Future resumes of new seed-7/99 jobs use their own exact contracts and the existing checkpoints.

Until the new runs finish, the live summaries report three completed seeds out of five and `incomplete: true`. Their means/SDs describe only the available three seeds, not a completed five-seed result. The saved pre-extension summary remains available separately.

## Preserved protocol

- Frozen input hashes, official seven labels, feature order/families, model settings, threshold 0.3, text preprocessing, checkpoint revision, and synthetic weighting are validated against accepted evidence.
- The prebuilt outer and inner source-group manifests are reused and validated; neither depends on the training seed. The dry run verifies 784 perfumes, 359 source groups, and five folds. `split_validation.json` records the fixed input hashes.
- Deep models train on inner-training rows only and use YouTube-only inner validation for early stopping. Synthetic reviews from validation/test perfumes are excluded. CPU execution preserves the accepted deep-model protocol.
- Classical primary models retain their accepted final fit on all outer-training rows. Their saved inner assignments describe the separate threshold diagnostic; they are not an additional holdout for the final classical fit. Weighted classical augmentation excludes outer-test perfumes. The repeat runner calculates the prespecified fixed-threshold primary metrics; it does not rerun secondary threshold tuning, bootstrap, or permutation diagnostics.
- Existing bootstrap, permutation, and data-selection seeds remain unchanged. No probabilities are averaged across training seeds. Each seed's Macro-F1, Macro-ROC-AUC, and macro average precision is calculated from its pooled five-fold OOF predictions using the existing metric implementation. Seed-level metrics are then summarized with mean and sample SD (`ddof=1`). Paired comparisons subtract metrics at matching seeds. Completed seed counts and incompleteness flags are explicit. Seed SD is not a bootstrap confidence interval.
- Standalone historical replay runners still enforce their original provenance gates. The new runner uses their validated inputs and training primitives with explicit training-seed overrides. `config/thesis_three_seed_code_revisions.json` permits only the recorded seed-handling code changes when deriving new loader protocols; data/config/checkpoint/result hashes are never relaxed. Other code changes invalidate resumption fingerprints.
- Main benchmark runners use `seeds.model_seed` for training and `seeds.data_seed` for splits, with backward-compatible fallback. Run IDs, manifests, comparison tables, and logs record both correctly. Use this orchestrator for isolated per-seed outputs and resumable execution.
- Separate DL ablations are outside this execution plan following the advisor's clarification about the text-only inputs.

## Pause, checkpoints, and progress

A graceful pause finishes the current deep epoch, saves its state, and exits. Resume begins with the next epoch. Checkpoints preserve model parameters, optimizer state, BERT scheduler state, Python/NumPy/PyTorch RNG state, DataLoader generator state, best model/epoch/validation scores, early-stopping counters, and epoch history. BiLSTM uses a constant learning rate and has no scheduler.

A forced termination or error during an epoch resumes from the last saved epoch boundary: **the unfinished epoch is repeated**. Before the first saved epoch, the first epoch repeats from deterministic initialization. Completed deep folds also have prediction checkpoints, so they are not retrained. Atomic checkpoint writes retain the preceding checkpoint if interrupted during a write. A dead process's job lock is recovered on resume; a live process's lock blocks a conflicting run.

Classical estimators have no epochs. They save completed fold-label fits and pause after the current fit. Resume skips these validated fit checkpoints; an interrupted, unsaved fit repeats. Optimizer/scheduler state is not applicable to those completed estimator fits.

The terminal and `outputs/thesis_three_seed_v1/status.json` receive updates at job/fit/epoch boundaries and a heartbeat every **30 seconds**, including model, condition, seed, fold, epoch, completed/total jobs, elapsed session time, and remaining-time estimate. Overall **job-completion percentage** is separate from **current training progress** and is explicitly not elapsed-time percentage. Deep progress uses fold and maximum-epoch budgets; early stopping can advance it abruptly. Classical progress counts completed fold-label fits. ETA uses observed completed-job runtimes of the same model/condition/variant and stays **“estimating”** until there are at least two observations for every remaining job type; copied historical runs and partial resumed runtimes do not calibrate it. Pauses do not interrupt a long fit or epoch before its safe checkpoint boundary.

The extension starts at **117/195 completed (60%)** after validating the old runs. Completed jobs are excluded from the execution queue and ETA from the outset. A dry run also refreshes status to show the new inventory, `status: ready`, and `training_launched: false`, replacing the obsolete 117/117 completion display.

## Verification

```sh
.venv/bin/python -m unittest discover -s src/training/tests -v
.venv/bin/python -m unittest src.training.tests.test_thesis_three_seed.TinyDeepResumeTests -v
.venv/bin/python -m src.training.run_thesis_three_seed --dry-run
```

Tests cover seed propagation, fixed splits, backward compatibility, distinct paths/IDs, incomplete/mismatched resumption, pooled per-seed metrics, sample SD, paired differences, status fields, synthetic exclusion, and real tiny interruption/resumption. Tiny BERT and BiLSTM pause/resume outputs and saved training states match uninterrupted execution exactly. A simulated failure after an optimizer step during BERT's second epoch proves that the unfinished epoch repeats and still reproduces uninterrupted predictions. Tiny classical fits prove saved tasks are skipped.

The original three-seed full-data runs are complete. No additional full-data training is started while verifying this extension. Tests cover five-seed selection, fixed splits, partial-summary flags, all-five-seed mean/sample SD, paired comparisons with missing seeds, preserving completed jobs, and progress from 60% to 100%. Tests intentionally adapt current-code fixtures through derived protocols; historical reports continue to be checked against their historical hashes.

## BERT control adapter repair (2026-09-29)

The first execution stopped before training BERT YouTube-only at seed 123: the shared weighted BERT helpers rejected that condition. Both split membership and class-balance helpers now support the control explicitly, using only natural inner-training chunks and the accepted perfume-level class weights. Augmented behavior is unchanged. New training also drops inherited seed-42 threshold-tuned columns rather than carrying them into another seed's predictions.

The 19 completed jobs were revalidated. Their predictions, metrics, completion records, and original code contracts remain unchanged. `config/thesis_three_seed_resume_compatibility.json` records exact audited old/new contract hashes for those 19 jobs and the one failed, unstarted BERT job. It permits no other configuration changes. The failed job has only a tokenizer cache; its original contract will be backed up when execution resumes. Partially trained deep checkpoints are not covered by this exception.

Regression tests now exercise the actual membership and class-weight helpers for both deep models under both training conditions, including tiny checkpoint pause/resume. Resume with:

```sh
.venv/bin/python -m src.training.run_thesis_three_seed --execute --resume --reuse-seed42
```

## Native BERT startup repair (2026-09-29)

The subsequent resume reached BERT control initialization but exited with a native segmentation fault before a training checkpoint was created. An isolated fresh-process probe reproduced exit code 139 when XGBoost was loaded before PyTorch. Loading PyTorch first passed the same real model initialization, forward/backward pass, and optimizer update, including the saved token cache and heartbeat setup. The runner now initializes PyTorch before XGBoost. No model, data, seed, package version, or hyperparameter was changed.

The regression suite includes a fresh Python subprocess that prepares all five models and runs one tiny batch through the actual cached BERT checkpoint. Earlier tiny-model tests could not expose this native initialization failure because their test process had already imported PyTorch.

The 19 completed jobs remain unchanged. The scoped compatibility registry retains the previous repair audit and binds the saved contracts to this startup-only change; it still rejects unrelated changes and partially trained deep checkpoints. The failed job's dead process lock is recovered by the existing resume logic. Full training was not restarted during diagnosis. To resume with native fault tracing enabled:

```sh
.venv/bin/python -X faulthandler -m src.training.run_thesis_three_seed --execute --resume --reuse-seed42
```
