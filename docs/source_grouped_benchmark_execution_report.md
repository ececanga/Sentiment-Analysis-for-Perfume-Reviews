# Phase 2 Source-Grouped Benchmark Execution Report

- Evidence cut-off: **2026-07-13 14:56:25 UTC**
- Experiment: `gemini_browser_og784_source_grouped_v1`
- Current status at cut-off: **completed and accepted; all five models, component-aware diagnostics, thesis exports, and the final audit gate passed**
- Post-acceptance Phase 3 supplement: **completed 2026-07-15; the accepted experiment root remains unchanged**

> **Defense boundary.** This report documents the completed leakage-repaired benchmark. The five-model comparison is accepted as final evidence for the narrowly defined exact-source-grouped evaluation. Acceptance does not convert the model-generated labels into human ground truth, establish channel-independent generalization, or remove the rare-label and giant-component limitations documented below.

## 1. Objective and scope

Phase 2 repairs a defense-critical weakness in the archived benchmark: perfumes that shared an exact YouTube video or normalized transcript could previously occur in different outer folds. The Phase 2 objective is to rerun the same seven-label, 784-perfume task while keeping each connected exact-source component atomic in both outer evaluation and inner validation.

The benchmark estimates agreement with the preserved Gemini browser annotation policy. It does **not** establish human olfactory-emotion ground truth. The project log records the labeler as Gemini 3.1 Pro, but the preserved browser responses do not contain machine-verifiable model-version or session metadata. The defense-safe claim is therefore that the models approximate a documented model-generated annotation policy.

All Phase 2 artifacts are written under:

`outputs/gemini_browser_og784_source_grouped_v1/`

The archived legacy root, `outputs/gemini_browser_og784/model_benchmark/`, remains historical, non-final evidence and is not overwritten.

## 2. Phase 1 archival anchor

Before leakage repair, the current-draft benchmark was frozen as an immutable, non-final snapshot.

| Item | Verified value |
|---|---|
| Snapshot ID | `thesis_current_draft_benchmark_2026-07-13` |
| Snapshot path | `archive/thesis_current_draft_benchmark_2026-07-13/` |
| Git tag | `v1.7-thesis-current-draft-snapshot` |
| Tagged commit | `af5f5dd7ced59481036901780549a73a3d8aae20` |
| Commit subject | `Archive current thesis benchmark before leakage repair` |
| Snapshot manifest SHA-256 | `8347446ec46ada63f8aaec8800c165eea752a3a23b5267b13a2648485d96e4c1` |
| Inventoried payload | 912 files; 247,901,161 bytes |
| Verification result | Passed on 2026-07-13: 912 files and 247,901,161 bytes verified |

Verification command:

```bash
.venv/bin/python scripts/freeze_thesis_current_draft_benchmark.py --verify-only
```

The Phase 2 command manifest also records `af5f5dd7ced59481036901780549a73a3d8aae20` as its base commit and records SHA-256 hashes for the active pipeline and training files. This is necessary because the source-group repair code was present as reviewed working-tree content after the Phase 1 commit; the command/stage receipts, rather than the base commit alone, identify the exact executed code.

## 3. Frozen Phase 2 inputs

The prospective configuration was fixed before the accepted source-grouped rerun. Its SHA-256 is `44852e49d8ab2654ed233fd48d0a237c3e6abb924331d38d259f5ba03792bff2`.

| Frozen input | Rows | SHA-256 |
|---|---:|---|
| `data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv` | 4,881 | `dc439566a60653939321fcd4b3cbe874320a436396530837cd1be24d84174782` |
| `labels/gemini_browser_og784/gemini_browser_labels_og784_v1.csv` | 784 | `d69318ea2ad04f574ccbe8d6b348041551a4762379673ec2e8cba0d2b0ad461e` |
| `outputs/gemini_browser_og784/analysis_out/gemini_browser_og784_feature_labels.csv` | 784 | `518048e34d1b9145cd98d70cf63dbda730b55e80eb7c66db42f3d73103c52976` |

The preparation audit reports that all three actual hashes matched their configured values before benchmark artifacts were written.

## 4. Source topology and grouped folds

### 4.1 Atomic source identity

`source_identity_v1` connects two perfumes when they share either:

- a valid, exact, case-sensitive 11-character YouTube video ID; or
- a SHA-256 hash of `Final Transcript` after HTML unescaping, Unicode NFKC normalization, case folding, replacement of non-alphanumeric runs, and whitespace collapse.

The atomic split unit is the undirected transitive connected component, stored as `source_group_id`.

| Topology quantity | Value |
|---|---:|
| Perfumes | 784 |
| Raw/resolved membership rows | 4,881 / 4,881 |
| Valid unique video IDs | 4,158 |
| Unique normalized transcript hashes | 4,155 |
| Shared video IDs | 392 |
| Shared normalized transcript hashes | 392 |
| Connected source components | 359 |
| Largest component | 334 perfumes (42.602% of the corpus) |

Component-size distribution:

| Component size | Number of components |
|---:|---:|
| 1 | 309 |
| 2 | 27 |
| 3 | 11 |
| 4 | 6 |
| 5 | 2 |
| 6 | 1 |
| 7 | 2 |
| 334 | 1 |

The 334-perfume component is kept intact. This satisfies leakage isolation but necessarily prevents balanced five-fold row counts and materially lowers the effective number of independent source groups.

### 4.2 Outer-fold sizes

| Fold | Test perfumes | Test source groups | Training perfumes | Training source groups |
|---:|---:|---:|---:|---:|
| 1 | 334 | 1 | 450 | 358 |
| 2 | 113 | 90 | 671 | 269 |
| 3 | 113 | 90 | 671 | 269 |
| 4 | 112 | 89 | 672 | 270 |
| 5 | 112 | 89 | 672 | 270 |

This is a grouped generalization test, not a balanced or multilabel-stratified split. Fold 1 tests one unusually large component; folds 2-5 test many small components. Pooled OOF metrics remain the prespecified headline calculation, but fold-level and component-aware sensitivity results are required for interpretation.

### 4.3 Manifest and audit hashes

| Artifact | SHA-256 |
|---|---|
| Source audit JSON | `e232f96b6ee42c8961b4037caa1b1b1cd50ee8b8595b564d84f2a9a346ac9d1a` |
| Outer fold manifest | `5338fbb3f05a51d2dc7ec42564f5cf9d31f3c742dc278e31fb1f4dbf5f47e579` |
| Shared inner manifest | `56172122cf0bc2d36e4ad5240feb3677eee27ab82387fc8fd0182f162bb24e0d` |
| Fold-balance table | `a1cc1a9974bfbc7f079446947ce4384240396155382bd84f7b6258eccba739c8` |

The completed source-audit gate independently checked 4,158 video sources, 4,155 transcript sources, 784 unique perfumes, 359 components, 3,136 inner-manifest rows, and 197 inner-validation assignments.

### 4.4 Legacy overlap repaired

| Diagnostic | Legacy cross-fold sources | Legacy affected perfumes | Phase 2 cross-fold count |
|---|---:|---:|---:|
| Valid video ID | 341 | 462 | 0 |
| `source_identity_v1` transcript hash | 341 | 462 | 0 |
| Exact/whitespace-normalized transcript variant | 340 | 459 | 0 under the Phase 2 normalization |

Across the legacy folds, 475 perfumes were affected by at least one shared exact source. The Phase 2 audit reports zero cross-fold video IDs, zero cross-fold normalized transcript hashes, and zero cross-fold source groups.

## 5. Metadata leakage audit and transcript-only amendment

Exact-source isolation does not make channel/reviewer metadata independent. The audit measured normalized metadata values that still occur across Phase 2 folds:

| Audit-only field | Unique normalized values | Cross-fold values | Affected perfumes |
|---|---:|---:|---:|
| Channel Title | 1,260 | 405 | 757 |
| Description | 3,504 | 55 | 151 |
| Title | 4,127 | 3 | 10 |

Connecting by channel would collapse nearly the full corpus and would answer a substantially different question. Instead, the pre-acceptance amendment made the deep-model input **transcript-excerpt only**:

- included in model text: `Final Transcript` excerpts;
- excluded from model text and retained only for audit: `Title`, `Description`, `URL`, and `Channel Title`;
- superseded title/description-inclusive deep-model attempts: not accepted as Phase 2 results.

The current deep-dataset summary explicitly records `model_input_fields: ["Final Transcript"]` and reports the transcript-only policy. The engineered-feature table likewise contains the declared NRC, VAD, Hartmann, Savani, GoEmotions, and Cardiff feature families rather than channel/title/description fields.

Residual caveat: transcript language can still encode reviewer style. The repaired claim is exact video/transcript isolation, not channel-independent generalization.

## 6. Trivial and prevalence baselines

All baseline probabilities are pooled out of fold and thresholded at the fixed threshold of 0.3. The train-fold prevalence baseline calculates each label probability from the complementary training folds only. Undefined ROC-AUC and average precision values are represented as null rather than silently dropped at the fold/per-label level.

| Baseline | Macro-F1 | Macro ROC-AUC | Macro average precision |
|---|---:|---:|---:|
| All-negative | 0.0000 | 0.5000 | 0.2808 |
| All-positive | 0.3523 | 0.5000 | 0.2808 |
| Train-fold prevalence | 0.3253 | 0.3930 | 0.2681 |

Baseline report SHA-256: `ba16797d040806369bfd362213a60a46831ceaf0e8347cfd911ecc35475f1503`.

The all-negative and all-positive baselines have identical ranking metrics because each assigns a constant score; they differ only after thresholding. The all-positive macro-F1 of 0.3523 is therefore the important trivial thresholded comparator. The average-precision baseline of 0.2808 is the mean label prevalence under constant ranking.

## 7. Final five-model pooled results

The table below reports pooled OOF metrics at the fixed threshold of 0.3. “Average precision” is used deliberately: the JSON field remains historically named `macro_pr_auc`, but it stores average precision.

| Model | Macro-F1 | ΔF1 vs all-positive | Macro ROC-AUC | ΔROC-AUC | Macro average precision | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.3592 | +0.0070 | 0.5473 | +0.0473 | 0.3145 | +0.0337 |
| Random Forest | 0.3306 | -0.0217 | 0.5638 | +0.0638 | 0.3346 | +0.0538 |
| XGBoost | 0.3260 | -0.0263 | 0.5683 | +0.0683 | 0.3094 | +0.0286 |
| BERT | 0.4029 | +0.0507 | 0.5533 | +0.0533 | 0.3590 | +0.0782 |
| BiLSTM + Attention | 0.3486 | -0.0036 | 0.4507 | -0.0493 | 0.2753 | -0.0055 |

Final pooled interpretation:

- BERT has the largest pooled point estimates for macro-F1 and macro average precision. XGBoost has the largest pooled macro ROC-AUC point estimate.
- Logistic Regression is the only other model above the all-positive macro-F1 comparator, and its margin is only 0.0070. Random Forest and XGBoost show ranking signal above the constant baseline but remain below the all-positive comparator at the fixed threshold.
- BiLSTM + Attention is below the all-positive macro-F1 comparator, the constant 0.5 ROC-AUC reference, and the constant-prevalence average-precision reference. This run provides no basis for a favorable BiLSTM performance claim.
- Ranking and thresholded classification conclusions remain separate. BERT's larger pooled point estimates are descriptive, not evidence of model superiority: all ten paired source-component bootstrap confidence intervals in Section 11 cross zero.

Every accepted model run contains 784 OOF rows and records the same configuration, outer-manifest, inner-manifest, and source-audit hashes. The three engineered-feature models use the frozen feature-label hash; BERT and BiLSTM + Attention use the frozen transcript-only JSONL hash.

## 8. Amended deep-text dataset

| Quantity | Verified value |
|---|---:|
| Perfume-level JSONL records | 784 |
| Chunk rows | 4,879 |
| Source transcript rows before deep deduplication | 4,881 |
| Duplicate transcript rows dropped | 2 |
| Source groups represented | 359 |
| Chunk count per perfume | min 1; median 7; mean 6.223; max 10 |

| Deep artifact | SHA-256 |
|---|---|
| `perfume_text_dataset_v1.jsonl` | `48f30d49041f80dc4bd8cfb755177b6b520c3b7fdb47dbf6cc66b243d97c129f` |
| `perfume_text_dataset_v1_chunks.csv` | `4e0bfaed7de1dd9eb3bd643f18b8589e720a80e7ce6eb6b991259703d4e00475` |
| `perfume_text_dataset_v1_summary.json` | `ebc8b5306247b1e110527b0e29f532a3cec7bebd767979440e887d71c82d18d8` |

The deep-dataset audit gate passed with 784 perfumes, 4,879 chunks, the frozen outer-manifest hash, zero cross-fold video IDs, and zero cross-fold transcript hashes.

## 9. Rare-label and validation-support caveats

The following supports are derived directly from the frozen feature-label table and the accepted outer/inner manifests. Vector positions correspond to outer folds 1-5.

| Label | Global positives | Outer-test positives by fold | Inner-validation positives by outer fold |
|---|---:|---|---|
| Joy | 622 | 247, 94, 103, 89, 89 | 39, 34, 30, 31, 37 |
| Sadness | 11 | 3, 1, 4, 2, 1 | 1, 1, 1, 0, 0 |
| Fear | 3 | 2, 0, 0, 0, 1 | 0, 0, 0, 0, 0 |
| Trust | 476 | 194, 74, 79, 68, 61 | 26, 25, 23, 25, 27 |
| Surprise | 44 | 20, 5, 5, 7, 7 | 2, 1, 3, 1, 3 |
| Anticipation | 366 | 155, 48, 47, 59, 57 | 15, 17, 17, 20, 13 |
| Disgust | 19 | 14, 0, 0, 3, 2 | 1, 1, 0, 1, 1 |

Defense implications:

- Fear is exploratory. Three of five outer test folds contain no positive fear examples, and none of the five inner-validation partitions contains a fear positive. Fear-specific inner threshold selection is therefore unsupported by positive validation evidence.
- Disgust has no outer-test positives in folds 2 and 3; sadness has no inner-validation positives for outer folds 4 and 5; disgust has none for outer fold 3.
- Per-fold ROC-AUC is undefined whenever a held-out fold contains only one class. Average precision is undefined when positive support is zero. These cells must remain null.
- Source isolation takes precedence over label balance. The giant component explains much of the outer-fold imbalance, so ordinary fold-to-fold standard deviations should not be interpreted as estimates from five exchangeable, similarly sized folds.
- The pooled seven-label macro metrics are reproducible summaries, but evidence for ultra-rare labels is not inferentially equivalent to evidence for joy, trust, or anticipation.

The completed `inner_fold_balance.csv` and `outer_fold_per_label_metrics.csv` artifacts formalize these checks. They contain 10 inner train/validation rows and 175 model-by-fold-by-label rows, respectively, retain undefined cells explicitly, and passed the diagnostics and final audit gates.

## 10. Acceptance gates and provenance status

| Stage/gate | Final status | Stage-receipt SHA-256 | Accepted evidence |
|---|---|---|---|
| Phase 1 archive verification | Passed | — | 912 files; 247,901,161 bytes |
| Source preparation | Completed | `fcbbd51e11722eadf451053f0ad7efc914121c18d4f08380087f15ad058186fb` | Membership, component, outer, inner, and fold-balance artifacts exist |
| Source audit gate | Passed | `0935479945a2a63b9761bbe70ac45b2680fe3134bf95a8d2f7aed81c5abd8db9` | 0 video, transcript, or component overlaps across outer folds |
| Dummy/prevalence baselines | Passed | `775fafd57b918da2c050de512a71e28065e129ce9c701200bc41b54a4a0980cc` | Three baseline rows; report hash verified |
| Logistic Regression smoke | Passed | `39e419c2dc63aa6ab3c07608c5ec0d1829f3017986cbd9366dcff976b05e6082` | 784 OOF rows; fixed-threshold metrics and shared provenance verified |
| Random Forest | Passed | `2403905a54e40eee7d294db90fadd0d8d1cd71af3e875ae1b286efdcb328fe4c` | 784 OOF rows; fixed-threshold metrics and shared provenance verified |
| XGBoost | Passed | `ae39f5653db4c7416f5da9f34c191ac2d61d97f6f9b2745949643afc968c13c2` | 784 OOF rows; fixed-threshold metrics and shared provenance verified |
| Transcript-only deep dataset | Completed | `0a1d620958aa02d70c913f72088c91acd99144ce9c5a28ee9c2d5b9950c99897` | Rebuilt after amendment; no superseded deep result accepted |
| Deep-dataset audit gate | Passed | `5bf8b3f2530b2222423e3b792683c73e205ef4e3a1cfe00b6d3a12c41c2568b5` | 784 JSONL rows; 4,879 chunks; exact-source checks passed |
| BERT | Passed | `2b1449234d70d4dd4589a8767d2d16c42039678a63e04f678bd913f29a5356f1` | 784 OOF rows; accepted at 2026-07-13 14:42:33 UTC |
| BiLSTM + Attention | Passed | `fe994a304cdeaaf80c65b64309423168267665581eb838497557e99e13c6fd41` | 784 OOF rows; accepted at 2026-07-13 14:50:29 UTC |
| Source-grouped/component diagnostics | Passed | `81788c2e8038d10775d0165617ee25bcbd1f9c8fba25e5c245936adf8f8fe065` | Fold support, component weighting, and 2,000-repetition paired bootstrap verified |
| Final thesis export | Passed | `1569537ddb6fa8aff2c20384d9cf089032ac3c298470999151931b8be2c4d355` | Five-row thesis table and four final figures verified |
| Final audit gate | Passed | `0cf8556d84a11393fc80e271695e7a6c50bda50f8e6af1e39ceff9555a1a3079` | All five models and all prerequisite receipts accepted at 2026-07-13 14:51:17 UTC |

The receipts record the base commit `af5f5dd7ced59481036901780549a73a3d8aae20`, configuration hash `44852e49d8ab2654ed233fd48d0a237c3e6abb924331d38d259f5ba03792bff2`, active pipeline/training source hashes, exact commands, stage fingerprints, expected outputs, and post-stage integrity checks. The final configuration's requirements for five model rows, shared outer and inner manifests, rare-label diagnostics, inverse-component-size weighting, paired component-cluster bootstrap differences, average-precision terminology, and a successful final audit were all satisfied without relaxation.

## 11. Final completion evidence

### 11.1 Accepted model artifacts

| Model | Metrics JSON SHA-256 | Run-manifest SHA-256 | OOF-prediction SHA-256 |
|---|---|---|---|
| Logistic Regression | `c7390d60506f63f3e24b13a5f4eb74baba00f6b57f413ff22b11125ed14698aa` | `d8cc11502a3c567e50d12645d26aae2bc80efca8d64b03ae9f4446645dac42c6` | `3b33ad2e529035c9312f32fdc5778f8cb7e598b0714cda6d5fa043f9866941f1` |
| Random Forest | `c298a975b1e9b10423651fdf86094740e08944883d6bb1a2721cb56c779bd6fa` | `bfa2b07632102ae9afea456b5e69ba12310148d3ee116defe70f015f426526ec` | `3d2d585adc52c91dc07e0cd0f52392426739a6bdb306a404b3be82ebd4af8e32` |
| XGBoost | `e389b6ae719486667e2726c8ff357b5d744b3ae015b8a19602341330179dfa0c` | `b0999cc4541cb634aa2290633e6211cb5b047d0eac67291889b56679cec89419` | `310055d3283c7c6ccf5b26b17516587e74b31d3dfc49b9d00cea911f3918b949` |
| BERT | `cf8a9189d7b637a135c7e8af313b440b2c58a13c8f40e87e0af642580d533314` | `5166e6eb7d2a2a8d5d5735495554d65dab7a5f784efa65c118afc385dde58520` | `3f41a0d90b60a964171571f671f5c53b738282fdbad4b546209ec061d9ce040d` |
| BiLSTM + Attention | `b3ffb60fcb58a07c06f2138958240d89fce2350c5a82f54bae206185e83a4248` | `6780aa778f265a8f4dc19833f324c7514a18f7e42deaf7b77327a8beb5090286` | `6bb2309a4080eaa602efd95293261d5c8c117a6393ea572856b95ad1bd72f624` |

BERT and BiLSTM + Attention both bind to the amended transcript-only dataset hash `48f30d49041f80dc4bd8cfb755177b6b520c3b7fdb47dbf6cc66b243d97c129f`. All five metrics files bind to the outer, inner, and source-audit hashes in Section 4.3.

### 11.2 Component-equal sensitivity and paired uncertainty

The component sensitivity gives every one of the 359 source components total weight one by assigning each perfume weight `1/source_group_size`. Macro metrics then average the seven labels without additional label weighting.

| Model | Component-equal macro-F1 | Component-equal macro ROC-AUC | Component-equal macro average precision |
|---|---:|---:|---:|
| Logistic Regression | 0.357544 | 0.589221 | 0.356434 |
| Random Forest | 0.341389 | 0.591946 | 0.360615 |
| XGBoost | 0.336111 | 0.591724 | 0.325914 |
| BERT | 0.483940 | 0.593236 | 0.466519 |
| BiLSTM + Attention | 0.352908 | 0.517747 | 0.294213 |

BERT has the largest component-equal point estimate for all three metrics. The paired source-group cluster bootstrap resampled the same 359 component draws for every model, used 2,000 repetitions with seed 42, and calculated 95% percentile intervals for component-equal macro-F1 differences.

| Model A | Model B | Point difference A − B | 95% bootstrap CI | Excludes zero |
|---|---|---:|---:|---|
| Logistic Regression | Random Forest | +0.016155 | [−0.016011, +0.046004] | No |
| Logistic Regression | XGBoost | +0.021433 | [−0.010301, +0.052883] | No |
| Logistic Regression | BERT | −0.126396 | [−0.153675, +0.028359] | No |
| Logistic Regression | BiLSTM + Attention | +0.004637 | [−0.014071, +0.026102] | No |
| Random Forest | XGBoost | +0.005278 | [−0.018530, +0.021634] | No |
| Random Forest | BERT | −0.142551 | [−0.172917, +0.025357] | No |
| Random Forest | BiLSTM + Attention | −0.011518 | [−0.030266, +0.016010] | No |
| XGBoost | BERT | −0.147829 | [−0.181923, +0.021250] | No |
| XGBoost | BiLSTM + Attention | −0.016797 | [−0.040753, +0.011324] | No |
| BERT | BiLSTM + Attention | +0.131033 | [−0.025937, +0.155591] | No |

All ten intervals cross zero. Consequently, the benchmark supports reporting ordered point estimates but **does not support a model-superiority claim** under the prespecified component-cluster uncertainty analysis.

| Diagnostic artifact | Rows | SHA-256 |
|---|---:|---|
| `inner_fold_balance.csv` | 10 | `639eb1a987f99ccf1b06673af58dad8462b852da781557f4a4dfbe111be69231` |
| `outer_fold_per_label_metrics.csv` | 175 | `5dfb21429c9b969ee5e15a8a071b754e1f25e1f5162f12c75caa93ac897c254b` |
| `component_weighted_model_metrics.csv` | 5 | `893ed0c2054208af220fe408493b12243d160c0febd088a4aaee3cfb8f8e0196` |
| `paired_component_bootstrap_macro_f1.csv` | 10 | `4ac112f2875282f7bbdcee8860b7a9794c9852bd775c69f86c05d50ab434ed01` |
| `source_grouped_diagnostics_report.json` | — | `f1c189bb14eff7ddb39c334575fa4995b8a39ad56771cf24897e05878e4475f2` |

### 11.3 Final thesis export and acceptance

| Final artifact | SHA-256 |
|---|---|
| `final_model_comparison_for_thesis.csv` | `0951b49f9236b67b88780e6684712e9ad64a465a498c7700961e832d2204d58d` |
| `all_models_macro_f1_comparison.png` | `4992c76e278997959289b46ffbaf8f799e8c8244b945e05d7140ccf396b3a263` |
| `all_models_macro_roc_auc_comparison.png` | `bde40039914f813495ce75387db0c22affa4ad581d332e15da9f44afbfe44e16` |
| `all_models_macro_pr_auc_comparison.png` | `cd2ee34c48d6cabacea49df3e2b597a31520f4e5b73d812724dd12239fcc67a2` |
| `all_models_grouped_metric_comparison.png` | `f0abe4b362f63782445bc34b5a582669c76bb4aefcb6e094123424633a374292` |
| `workflow/command_manifest.json` | `81d9aad22c3c99cc0f8b802b84096a95be525976c178dd32cefcd187a8ed0c7f` |
| `workflow/run_status.json` | `b9de48071e6ec00c84fc9f358035eb362e0b2fd6033bdc300a3f03dbf08a600f` |
| `workflow/stages/final_audit_gate.json` | `0cf8556d84a11393fc80e271695e7a6c50bda50f8e6af1e39ceff9555a1a3079` |

The final audit receipt has `status: "completed"`, `checks.status: "passed"`, and explicitly lists all five accepted model IDs. A post-gate independent verification additionally confirmed exact binary OOF truth/prediction cells before integer conversion, finite probabilities in `[0,1]`, exact fixed-threshold decisions, paired identities across all five OOF files, numeric equality of the rounded thesis export with the source comparison table, and byte-identical regeneration of the 784-record/4,879-chunk transcript-only dataset from the frozen inputs. Its SHA-256 is `955beed83565cd6e7527dd70d7d52f8231eba06fe6d2a4fe82c49a205265d7b8`.

Every regular file below the experiment root, except the checksum file itself, is anchored by the 95-entry `SHA256SUMS` manifest. The manifest passed full verification and has SHA-256 `11cca80d9767e0e368c2502aae57f7263c0b5d9bfd20794f85df58681f47f852`. **Final benchmark acceptance is granted for the defense-bounded claim stated in this report.**

### 11.4 Phase 3 post-acceptance diagnostics

Phase 3 adds versioned diagnostics without modifying the accepted benchmark
root or its receipts. The six-rule baseline supplement repeats the three
accepted rules and adds always joy/trust/anticipation, deterministic uniform
random scores, and deterministic train-prevalence-stratified random
predictions.

| Non-learned rule | Macro-F1 | Macro ROC-AUC | Macro average precision |
|---|---:|---:|---:|
| All-negative | 0.0000 | 0.5000 | 0.2808 |
| All-positive | 0.3523 | 0.5000 | 0.2808 |
| Always joy, trust, and anticipation | 0.3253 | 0.5000 | 0.2808 |
| Train-fold prevalence | 0.3253 | 0.3930 | 0.2681 |
| Uniform-random scores | 0.3025 | 0.4830 | 0.2824 |
| Stratified-random predictions | 0.2895 | 0.5010 | 0.2806 |

The all-metric supplement reuses the accepted five OOF matrices and the same
359 component draws for all models, with 2,000 repetitions and seed 42. Its
macro-F1 rows reproduce the accepted paired table exactly. It additionally
reports model-level intervals and paired differences for macro average
precision and macro ROC-AUC. All ten macro-F1 intervals cross zero. Five
average-precision intervals exclude zero: Logistic Regression, Random Forest,
XGBoost, and BERT each exceed BiLSTM, and Random Forest exceeds XGBoost. Two
ROC-AUC intervals exclude zero: XGBoost and BERT each exceed BiLSTM. BERT's
average-precision differences from the three classical models all cross zero,
so the primary ranking analysis does not establish BERT as an overall winner.

| Supplemental artifact | SHA-256 |
|---|---|
| `source_grouped_extended_baselines_v1/extended_baseline_summary.csv` | `4e079de861d8ece029f155ccd285d6adaad87ab0ca0c488c20f3c27934a0c501` |
| `source_grouped_extended_baselines_v1/REPORT.json` | `986460558af67a68142a03fe3214242e3e5b2167ad2ac99495e5135c2ffcedf2` |
| `source_grouped_all_metric_bootstrap_v1/model_metric_bootstrap_intervals.csv` | `653716822e48de0c9de8a151c745e190151f8cba2c87bf8c4dfa4dcde7697dda` |
| `source_grouped_all_metric_bootstrap_v1/paired_model_metric_bootstrap_differences.csv` | `adda4bc734e82364a90fac7e9be80bf08c3758fccb1d05e2982d6587619832ad` |
| `source_grouped_all_metric_bootstrap_v1/REPORT.json` | `7f2b025b293b72e2091a2ec6cde710c2b027806ef57242750f17b42a8b300320` |

### 11.5 Phase 5 supervised feature-family analysis

Phase 5 reuses the final class-weighted Logistic Regression pipeline and the
frozen source-grouped outer folds to compare 15 representations: the full
58-feature hybrid, eight family-only or grouped variants, and six
leave-one-family-out variants. The full-hybrid reproduction gate passed with
zero truth or prediction mismatches and a maximum probability difference of
`1.11e-16`. All 525 label-fold fits completed without convergence warnings.
The main thesis reports only the full model and six leave-one-family-out rows;
the family-only, permutation, coefficient, and complete bootstrap results are
kept here and in the versioned supplement to avoid overloading the central
defense narrative.

The pooled point estimates do not support uniform superiority of the full
fusion. Hartmann-only has the largest seven-label macro average-precision point
estimate (`0.3565` versus `0.3145` for the full hybrid), but this ordering is
driven strongly by fear, which has only three positives. When fear is excluded,
the full hybrid is `0.3616`, Hartmann-only is `0.3581`, and full minus
GoEmotions is the pooled point leader at `0.3812`. Under component-equal
six-label weighting, the full hybrid exceeds Hartmann-only by `0.0368`, with a
95% interval of `[0.0021, 0.0906]`. Hartmann-only is therefore not treated as
an overall winner.

Hartmann has the clearest fear-robust leave-one-family-out signal. The
component-equal full-minus-no-Hartmann macro average-precision difference is
`0.0217 [0.0037, 0.0462]` with all seven labels and
`0.0180 [0.0003, 0.0383]` without fear. The seven-label GoEmotions and Cardiff
leave-out intervals exclude zero, but their fear-excluded intervals do not.
No full-versus-variant macro-F1 interval excludes zero. All 14 contrasts are
unadjusted post-hoc comparisons, and the bootstrap conditions on the fitted
OOF models, so these findings remain exploratory.

Thirty held-out family-permutation repetitions and foldwise standardized
coefficient diagnostics support the same bounded interpretation. Hartmann's
mean macro-average-precision drop remains positive after fear is excluded
(`0.0075` pooled and `0.0274` component-equal). Savani changes from a negative
seven-label mean drop to a positive fear-excluded mean drop, so it is described
as label-sensitive or potentially redundant rather than harmful. Median signed
coefficient-vector Spearman correlations range from `0.677` to `0.888`, while
median top-10 Jaccard overlap ranges from `0.250` to `0.538`. Coefficients and
permutations describe predictive reliance under correlated features, not
causal importance.

| Phase 5 artifact | SHA-256 |
|---|---|
| `source_grouped_feature_ablation_v1/ablation_summary.csv` | `ae7c14ce92347b2d9301c756bfefe5f71c59a275347ddd18dd380d049c9e4364` |
| `source_grouped_feature_ablation_v1/paired_bootstrap_vs_full.csv` | `ea0f0069ec22bcb673bfee1ae90ca27f6129404c3586be067c60dac2d5b46bd3` |
| `source_grouped_feature_ablation_v1/fear_excluded_ablation_summary.csv` | `7d574ae70cd5fcacaf60bb7c360ff36f3ff020c3a580fa3afb43147dac135327` |
| `source_grouped_feature_ablation_v1/fear_excluded_paired_bootstrap_vs_full.csv` | `94ed2cc45470b9b9c89777ed7337fce2d6549fc5e987150e41ab71643059e458` |
| `source_grouped_feature_ablation_v1/family_permutation_importance.csv` | `c88dbe8f71413f0d23a607c1d69a9ae4aa0db5ae0258906e6d226330fb14b01c` |
| `source_grouped_feature_ablation_v1/fear_excluded_family_permutation_importance.csv` | `7a05b36ce04226d239ad8704f0f3ca539d64918a2fdd4d0a01dc2bee320b566f` |
| `source_grouped_feature_ablation_v1/coefficient_vector_stability.csv` | `d05799d57abe2178a801c3a3976c432fbf397e1baaecbffc2da81bfb573b6cc5` |
| `source_grouped_feature_ablation_v1/REPORT.json` | `43a31dc31ad5ea04930a2dce6d37bc249707dbde014a817a11b3352d503bddb2` |
| `source_grouped_feature_ablation_v1/SHA256SUMS` | `929baf572d87e5bc1316e978a2d1fb4ca1bdaa661b96c6cb2e463876d28bc5fd` |

### 11.6 Phase 6 weighted synthetic-augmentation LR contrast

Phase 6 replaces the earlier Logistic Regression augmentation comparison with
a prespecified source-grouped contrast. Both conditions use the accepted
class-weighted Logistic Regression, the same frozen 58 engineered features,
and the same outer folds. In the augmented condition, each real YouTube
perfume has training weight one and its three synthetic reviews have weight
one third each, so the synthetic block cannot receive three times the training
influence merely because it has three rows. These weights are applied to both
feature standardization and classifier fitting. Synthetic rows matching any
held-out perfume are excluded, and evaluation is performed only on the 784
real held-out YouTube perfumes.

| Condition | Pooled macro-F1 | Pooled macro average precision | Pooled macro ROC-AUC |
|---|---:|---:|---:|
| YouTube only | 0.3592 | 0.3145 | 0.5473 |
| YouTube + weighted synthetic x3 | 0.3737 | 0.3505 | 0.5372 |

The paired component bootstrap uses the same 359 source-component draws for
both conditions, with 2,000 repetitions and seed 42. Positive differences
favor augmentation.

| Label scope | Metric | Augmented − control | 95% component interval | Excludes zero |
|---|---|---:|---:|---|
| Seven labels | Macro-F1 | +0.0227 | [+0.0011, +0.0466] | Yes |
| Seven labels | Macro average precision | +0.1202 | [−0.0366, +0.1402] | No |
| Seven labels | Macro ROC-AUC | +0.0081 | [−0.0195, +0.0358] | No |
| Fear excluded | Macro-F1 | +0.0159 | [−0.0037, +0.0338] | No |
| Fear excluded | Macro average precision | −0.0164 | [−0.0516, +0.0099] | No |
| Fear excluded | Macro ROC-AUC | +0.0049 | [−0.0273, +0.0367] | No |

The seven-label macro-F1 result is positive, but the broader evidence does not
establish a robust general augmentation benefit. The primary macro-average-
precision interval includes zero, and fear has only three positive real
examples. Fear's component-equal average precision changes from 0.0480 to
0.9882; after fear is removed, the average-precision point difference is
slightly negative and all three intervals include zero. The defensible Phase 6
conclusion is therefore that weighted synthetic augmentation produced some
better point estimates, but its benefit is uncertain and highly sensitive to
an ultra-rare label. The experiment remains a mixed-granularity sensitivity
because real rows are perfume aggregates while synthetic rows are individual
generated reviews.

The validation gate found zero cross-fold source components, zero held-out
synthetic overlap, and exact fixed-threshold agreement between the accepted
control OOF predictions and the control replay. The largest replay probability
difference was `8.03e-05`, within the prespecified `1e-4` tolerance. All five
new tests and 22 selected source-grouped regression tests passed, and a full
rerun reproduced the output manifest byte for byte.

| Phase 6 artifact | SHA-256 |
|---|---|
| `config/evaluation_config_source_grouped_weighted_synthetic_lr_v1.yaml` | `8091a6ea23c1d9b4061c72d1550e7524635b2cd747b0ea2cebd9b2b3bbe4b09f` |
| `source_grouped_weighted_synthetic_lr_v1/condition_summary.csv` | `8f85156012388cf9ef8a19d2f337404a1ab235682323ad9fc84cb614f5492e63` |
| `source_grouped_weighted_synthetic_lr_v1/paired_component_bootstrap_differences.csv` | `3dd352dd0e641bebffc6c865932ee8617b34b5f02713970c4c48c70aa96b00a6` |
| `source_grouped_weighted_synthetic_lr_v1/REPORT.json` | `858846deec6ddbf5c40aabdfc0103263734155ec5407ce7f81e5f80568816cb0` |
| `source_grouped_weighted_synthetic_lr_v1/SHA256SUMS` | `adf975d439804fb183a6e5d2850c9ba0bff0c62293422f32808e71014904f278` |

## 12. Defense-ready final conclusion

Phase 2 corrected the demonstrated exact-source leakage in the archived perfume-level folds. All five accepted models use the same source-component outer folds, the same source-safe inner manifest, the same frozen inputs appropriate to their model family, and independently checked OOF provenance. The deep models use only transcript excerpts after the pre-acceptance metadata amendment.

At the pooled perfume level, BERT has the largest primary macro average-precision (0.3590) and primary thresholded macro-F1 (0.4029) point estimates, while XGBoost has the largest secondary macro ROC-AUC point estimate (0.5683). With every source component given equal total weight, BERT has the largest point estimate for all three metrics. These rankings are descriptive: all ten paired macro-F1 confidence intervals cross zero, and BERT's primary ranking differences from the three classical models also cross zero. The expanded bootstrap supports a few narrower ranking differences, mostly against BiLSTM, but does not justify declaring one universally superior model.

The baseline comparison also calls for restraint. Only BERT and Logistic Regression exceed the all-positive macro-F1 comparator, and Logistic Regression's margin is 0.0070. Random Forest and XGBoost show ranking signal but do not beat that comparator at the fixed threshold. BiLSTM + Attention does not exceed the all-positive macro-F1, 0.5 ROC-AUC, or prevalence average-precision references in this run.

The result remains bounded by the 334-perfume component, the nonexchangeable fold structure, 359 effective exact-source components, and very low support for fear and other rare labels. It estimates agreement with a documented model-generated annotation policy, not human olfactory-emotion truth. Exact video/transcript isolation also does not establish channel- or reviewer-independent generalization because transcript language can preserve reviewer style.

Within those limits, the benchmark is **defense-ready and artifact-complete**: its five-model table, component-aware sensitivity analysis, uncertainty result, and final acceptance gate may replace the superseded Phase 1 comparison in Chapters 4 and 5. The defensible claim is improved exact-source-isolated agreement estimation, not universal model superiority or external human validity.

One reproducibility qualification remains explicit. The accepted BERT runner used `local_files_only: true` and recorded the repository `monologg/bert-base-cased-goemotions-original`, but did not prospectively pass an immutable Hugging Face revision. The cache inspected immediately after the run resolved `main` to revision `13c44c849132f82bb61188d909a574badffb27a3`; the configuration, tokenizer, vocabulary, and 433,412,096-byte model-weight hashes are preserved in `bert_checkpoint_provenance.json` (SHA-256 `9858cc2ba3a9032493b2ffb502607c2cd260b021bf993f8ae185d7e19795fc3c`). This is supporting post-run provenance, not a claim of prospective revision pinning. A future rerun should pass that revision explicitly.

## 13. Primary evidence artifacts

- Configuration: `config/evaluation_config_gemini_browser_og784_source_grouped_v1.yaml`
- Command manifest: `outputs/gemini_browser_og784_source_grouped_v1/workflow/command_manifest.json`
- Workflow status: `outputs/gemini_browser_og784_source_grouped_v1/workflow/run_status.json`
- Complete artifact checksums: `outputs/gemini_browser_og784_source_grouped_v1/SHA256SUMS`
- Supplemental final verification: `outputs/gemini_browser_og784_source_grouped_v1/workflow/supplemental_final_verification.json`
- Stage receipts: `outputs/gemini_browser_og784_source_grouped_v1/workflow/stages/`
- Source audit: `outputs/gemini_browser_og784_source_grouped_v1/source_audit/source_group_audit.json`
- Legacy overlap diagnostic: `outputs/gemini_browser_og784_source_grouped_v1/source_audit/legacy_fold_overlap_diagnostic.json`
- Fold support: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/fold_balance.csv`
- Baselines: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/dummy_prevalence_baselines/`
- Extended six-rule baselines: `outputs/source_grouped_extended_baselines_v1/`
- All-model metrics: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/metrics/`
- All-model OOF predictions: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/predictions/`
- Deep summary: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/datasets/perfume_text_dataset_v1_summary.json`
- BERT checkpoint cache provenance: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/logs/bert_checkpoint_provenance.json`
- Inner-fold balance: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/inner_fold_balance.csv`
- Outer-fold per-label metrics: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/outer_fold_per_label_metrics.csv`
- Component-equal metrics: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/component_weighted_model_metrics.csv`
- Paired component bootstrap: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/paired_component_bootstrap_macro_f1.csv`
- All-metric component bootstrap: `outputs/source_grouped_all_metric_bootstrap_v1/`
- Supervised feature-family analysis: `outputs/source_grouped_feature_ablation_v1/`
- Weighted synthetic-augmentation LR contrast: `outputs/source_grouped_weighted_synthetic_lr_v1/`
- Diagnostics provenance report: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/source_grouped_diagnostics_report.json`
- Final thesis comparison: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/final_model_comparison_for_thesis.csv`
- Final five-model figures: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/figures/all_models_*.png`
- Final audit receipt: `outputs/gemini_browser_og784_source_grouped_v1/workflow/stages/final_audit_gate.json`
- Phase 1 archive manifest: `archive/thesis_current_draft_benchmark_2026-07-13/experiment_manifest.json`
