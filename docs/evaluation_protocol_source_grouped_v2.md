<!--
Date Created: 2026-07-13
Date Last Modified: 2026-07-13
-->

# Source-Grouped Evaluation Protocol v2

## Status and scope

This protocol prespecifies the leakage-repaired evaluation of the 784-perfume
Gemini-browser branch. It does not retroactively validate the results under
`outputs/gemini_browser_og784/model_benchmark`. Those results remain archived,
non-final current-draft evidence.

The new experiment is isolated under:

- experiment root: `outputs/gemini_browser_og784_source_grouped_v1/`
- benchmark root:
  `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/`
- configuration:
  `config/evaluation_config_gemini_browser_og784_source_grouped_v1.yaml`

Nothing in the old result root may be overwritten or copied forward as if it
had been produced by the new split. Model metrics become candidate final
evidence only after every preflight and artifact-contract check in this
document passes.

## Why the split is being replaced

The previous shared five-fold manifest grouped only by `perfume_id`. A source
audit found that source material was reused across different perfume rows:

- 392 video IDs were associated with more than one perfume;
- 341 of those video IDs crossed the old folds, affecting 462 perfumes; and
- the earlier exact/whitespace-normalized transcript audit found 340 identities
  crossing the old folds, affecting 459 perfumes; and
- applying the stricter `source_identity_v1` punctuation-collapsing transcript
  normalization used by this protocol identifies 341 cross-fold transcript
  identities affecting 462 perfumes. The additional identity is a
  punctuation-only transcript variant.

When two perfume examples contain evidence from the same video or transcript,
putting one in training and the other in testing allows source-specific wording,
speaker style, and recording context to cross the evaluation boundary. A
perfume-only split therefore tests more than generalization to unseen source
material. The v2 split makes the connected source component, rather than the
individual perfume row, the indivisible split unit.

## Frozen inputs

The following files are the only canonical inputs for the v2 split and rerun.
SHA-256 is calculated over the exact file bytes, before parsing.

| Input key | Path | Rows | SHA-256 |
| --- | --- | ---: | --- |
| `raw_csv` | `data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv` | 4,881 | `dc439566a60653939321fcd4b3cbe874320a436396530837cd1be24d84174782` |
| `labels_csv` | `labels/gemini_browser_og784/gemini_browser_labels_og784_v1.csv` | 784 | `d69318ea2ad04f574ccbe8d6b348041551a4762379673ec2e8cba0d2b0ad461e` |
| `feature_label_csv` | `outputs/gemini_browser_og784/analysis_out/gemini_browser_og784_feature_labels.csv` | 784 | `518048e34d1b9145cd98d70cf63dbda730b55e80eb7c66db42f3d73103c52976` |

The split builder and every model runner must fail closed if the relevant hash
does not match the configuration. A hash change defines a different dataset
version and requires a new configuration and output root; updating the stored
digest in place is not an acceptable repair.

## `source_identity_v1`

### Canonical perfume resolution

The raw `Perfume` value is converted with the repository's existing
`normalize_perfume_id` rule. Every resolved ID must occur in both 784-row
canonical tables. An unresolved ID is an error, not a row to discard silently.

### Source keys

Two source keys are constructed independently for each raw review row.

1. Video identity
   - strip surrounding whitespace from `Video ID`;
   - retain case;
   - accept only an 11-character ID matching `[A-Za-z0-9_-]{11}`; and
   - ignore an empty or invalid value as a source key while counting it in the
     audit.
2. Transcript identity
   - read the complete `Final Transcript`, not a model excerpt;
   - HTML-unescape it;
   - normalize Unicode with NFKC;
   - apply Unicode case folding;
   - replace each punctuation or non-alphanumeric run with one space;
   - collapse whitespace and strip the result; and
   - SHA-256 hash its UTF-8 bytes.

An empty normalized transcript does not become a shared key. This prevents all
missing transcripts from being incorrectly joined into one component.

### Component construction

Treat each canonical `perfume_id` as a graph node. Connect two nodes when they
share at least one valid video ID or normalized transcript hash. Take the
undirected transitive closure: if A shares a source with B and B shares another
source with C, A, B, and C belong to one component even when A and C have no
direct key in common.

Each component receives a deterministic SHA-256-derived `source_group_id` from
the grouping version, sorted member IDs, and sorted source keys. Component IDs
must not depend on input row order. Duplicate raw membership rows are
deduplicated for graph construction and counted in the audit.

The required source audit is:

- `source_audit/source_memberships.csv`
- `source_audit/source_components.csv`
- `source_audit/source_group_audit.json`

The supplementary
`source_audit/legacy_fold_overlap_diagnostic.json` records both the archived
340/459 exact-transcript result and the 341/462 result under
`source_identity_v1`, so the normalization change is explicit rather than
appearing as an unexplained discrepancy.

The JSON report must record the grouping version, input paths and hashes,
normalization rules, input and resolved row counts, source-key counts,
component-size distribution, largest component, and all validation outcomes.

## Build once, then reuse

The manifests are created in a distinct preflight step:

```bash
python -m src.training.source_grouped_folds \
  --config config/evaluation_config_gemini_browser_og784_source_grouped_v1.yaml \
  --raw-csv data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv \
  --labels-csv labels/gemini_browser_og784/gemini_browser_labels_og784_v1.csv \
  --feature-label-csv outputs/gemini_browser_og784/analysis_out/gemini_browser_og784_feature_labels.csv \
  --out-root outputs/gemini_browser_og784_source_grouped_v1 \
  --n-splits 5 \
  --inner-n-splits 10 \
  --seed 42
```

The required split artifacts are:

- `model_benchmark/tables/fold_assignments.csv`
- `model_benchmark/tables/inner_validation_folds.csv`
- `model_benchmark/tables/fold_balance.csv`

Model runners are consumers of these files. They must not create a missing
source-grouped manifest, silently fall back to ordinary KFold, or modify a
manifest after one model has run. Every model run records the SHA-256 hashes of
the shared manifests and source audit so equality can be checked across all
five models.

## Outer evaluation

The outer design is labelled **5-fold source-component grouped CV**.

- unit reported in OOF output: one `perfume_id`;
- hard grouping unit: one `source_group_id`;
- number of folds: 5;
- one prebuilt manifest shared by Logistic Regression, XGBoost, Random Forest,
  BERT, and BiLSTM + Attention; and
- hard rule: every source component occurs in exactly one test fold.

Source isolation is a hard constraint. Approximate fold size and label support
are secondary objectives and may never be improved by cutting a component.
The specified `GroupKFold` assignment directly balances perfume-row counts;
label support is an audited desideratum, not an additional stratification
algorithm or a basis for post-hoc fold swapping.
`fold_balance.csv` must report row count, source-group count, and positive
support for every official label in every test fold and corresponding training
partition.

The primary metrics are computed once from the pooled 784-row OOF matrix. This
avoids giving a small fold and a very large fold equal weight. Fold-level
metrics are diagnostics, not the primary estimator.

## Giant-component risk

A preliminary reconstruction from the frozen source relationships produced 359
connected components, with one component containing 334 of 784 perfumes
(42.6%). The authoritative counts are those written by
`source_group_audit.json`, but this preflight result identifies a structural
risk that must be disclosed.

A 334-row atomic component is larger than the ideal 156.8-row fifth of the
dataset. Five balanced folds are therefore impossible under strict source
isolation. Expected consequences include:

- one test fold much larger than the others;
- different label prevalence across folds;
- a much smaller effective number of independent source units than 784; and
- unstable fold-level dispersion estimates.

The giant component must remain intact. It is not defensible to split it merely
to obtain visually balanced folds. The audit, fold sizes, component counts, and
per-fold supports must be reported next to the benchmark. The thesis must call
the design source-isolated but imbalanced, rather than stratified or balanced.

An ordinary standard deviation across five unequal folds is not a confidence
interval. Any later bootstrap or paired uncertainty analysis must resample at
the `source_group_id` level and explain that a component containing roughly 43%
of observations limits what such resampling can establish.

## Inner validation and threshold selection

For each outer fold, inner validation is derived only from that fold's outer
training components. The shared
`model_benchmark/tables/inner_validation_folds.csv` uses
`deterministic_source_group_holdout` with seed 42.

The builder applies `GroupKFold` to the outer-training source groups with up to
10 candidate folds. It selects the candidate whose perfume-row count is
closest to one tenth of the outer-training rows; ties prefer the smaller
candidate and then a SHA-256 ordering derived from seed 42, the outer-fold ID,
and candidate ID. This keeps the 334-row component in inner training instead
of allowing it to become a disproportionately large validation partition,
without ever dividing that component.

Hard rules are:

- no outer-test perfume or source group appears in inner training or
  validation;
- a source group cannot cross inner training and validation;
- early stopping, model selection, preprocessing learned from data, and
  threshold selection use inner-training/validation data only; and
- all five models use the same recorded inner roles wherever the protocol
  requires an inner partition.

The fixed threshold 0.3 remains the prespecified main-table decision rule.
Global and per-label threshold tuning are secondary diagnostics. The held-out
outer test fold may not influence either choice. If a rare label has no
positive example in an inner validation partition, that partition contains no
evidence for tuning that label: the closest-to-0.3 tie rule acts as the
deterministic fallback, and the zero support must be visible in the fold audit.

## Rare-label feasibility and interpretation

The frozen positive supports are:

| Label | Positive rows |
| --- | ---: |
| joy | 622 |
| sadness | 11 |
| fear | 3 |
| trust | 476 |
| surprise | 44 |
| anticipation | 366 |
| disgust | 19 |

With only three fear-positive perfumes and five outer folds, at least two test
folds must have zero fear positives even before source grouping is considered.
Grouping can concentrate multiple positives in one component and make the
imbalance worse. Multilabel stratification cannot override source isolation.

Consequently:

- pooled OOF F1, ROC-AUC, and average precision are the official estimates;
- per-fold ROC-AUC is undefined when a fold contains only one true class;
- per-fold average precision is marked undefined when there are no positives;
- undefined results remain `NA`/`null` and are never converted to zero or
  silently removed without reporting the changed denominator;
- train and test support accompanies every per-label result;
- a one-class outer training partition uses only an explicitly logged constant
  fallback, never an unreported estimator substitution; and
- fear is exploratory and cannot support a general model-ranking claim.

Macro-F1 gives rare labels equal numerical weight, but it does not make three
positive examples reliable. Macro average precision must be called **average
precision**, not trapezoidal PR-AUC, because the implementation uses
`average_precision_score`.

## Models, seeds, and comparable outputs

The prespecified model set remains:

1. Logistic Regression
2. XGBoost
3. Random Forest
4. BERT
5. BiLSTM + Attention

Data and model seeds remain 42. There is no best-seed selection, repeated CV,
or model-specific outer split. Deep-model text data must be rebuilt under the
new benchmark root so its embedded `fold_id` values match the v2 manifest. The
old JSONL with old fold IDs is forbidden as a v2 model input.

### Pre-acceptance audit amendment: transcript-only deep input

Before any deep-model run was accepted, a 2026-07-13 scientific audit found
that 55 exact normalized descriptions crossed the new outer folds, affecting
151 perfumes. Title, description, URL, and channel fields can contain repeated
reviewer boilerplate not covered by the exact video/transcript grouping rule.
The accepted Phase 2 JSONL therefore uses only excerpts from `Final Transcript`
as `chunk_text`; those metadata fields remain available for audit but cannot be
model input. The interrupted title/description-inclusive BERT attempt produced
no accepted benchmark row.

This repairs the direct repeated-metadata pathway but does not claim
reviewer-independent generalization. Grouping by channel would connect nearly
the full corpus, and reviewer style can remain in transcript language. The
defense-safe claim is exact video/transcript isolation, accompanied by
component-aware sensitivity analysis—not complete domain independence.

Required metrics are macro-F1 at threshold 0.3, macro ROC-AUC, and macro average
precision from pooled OOF predictions over all seven official labels. Per-label
support, precision, recall, F1, ROC-AUC, and average precision are required
diagnostics. The historical artifact key `macro_pr_auc` may be retained for
compatibility, but tables and prose must label the quantity macro average
precision.

Three trivial diagnostics are required beside, but not mixed into, the five
prespecified model rows: all-negative, all-positive, and held-fold predictions
equal to each label's complementary-training-fold prevalence. These use the
same fixed threshold of 0.3. They establish whether learned models improve on
label prevalence; in particular, model-superiority language is not justified
by a score that merely matches the always-positive rule.

Because one connected component contains 334 of 784 perfumes, pooled OOF
metrics remain the prespecified primary result but do not represent 784
independent sources. The final diagnostic bundle must also report metrics with
each source component given equal total weight and deterministic paired
component-cluster bootstrap intervals for model differences. These sensitivity
results govern superiority claims; ordinary row bootstrap and five-fold
standard deviations are not valid substitutes.

## Acceptance checks

The split is ready only when all of the following hold:

- all three frozen hashes match;
- all 784 canonical perfume IDs occur exactly once in the outer manifest;
- fold IDs are exactly 1 through 5;
- every outer `source_group_id` occurs in exactly one fold;
- every inner row belongs to the corresponding outer-training set;
- no inner source group crosses `train` and `val` for an outer fold;
- the source audit and fold-balance report disclose the giant component and
  rare-label supports; and
- deep `chunk_text` contains transcript excerpts only and embedded official,
  excluded, and VAD labels exactly match the frozen label table;
- the old result root is unchanged.

A model row is complete only when its run manifest, OOF predictions, metrics,
per-label table, comparable plots, log, and interpretation note exist and all
record the same input and split provenance. The main comparison table is
candidate final evidence only after all five prespecified rows are complete.
The acceptance gate recomputes fixed-threshold pooled metrics and per-label
values from OOF predictions, verifies `pred = (probability >= 0.3)`, requires
the plots/log/note files, and checks the comparison row numerically. Outer-fold
metrics serialize ROC-AUC as undefined unless both classes are present and
average precision as undefined when positive support is zero. Inner validation
support is reported separately; threshold optimization remains only a
secondary sensitivity analysis and is not probability calibration.

## Defense-safe description

The repaired evaluation can be described as follows:

> All perfumes connected through a shared YouTube video or normalized full
> transcript were assigned to the same outer fold. This prevents the same
> source material from appearing in both training and testing. The constraint
> creates a large connected component and unequal fold sizes, while very rare
> labels remain statistically unstable; therefore results are reported from
> pooled out-of-fold predictions with component and label-support diagnostics.
