# Thesis Pipeline Design Report

Date prepared: 2026-06-08  
Repository: `/Users/ece/Desktop/EceFolder/Sentiment Analysis`  
Purpose: summarize the full thesis pipeline, the design choices, the Gemini Flash vs Pro decision, the Phase2A expansion pause, NRC/VAD placement, ablation work, benchmark outputs, and current scope boundaries.

## Executive Summary

This thesis builds an applied NLP pipeline for emotional analysis of perfume reviews. The central data source is YouTube perfume review transcripts, because the thesis needs reviewer language that contains emotional tone. Product descriptions and note pyramids were explicitly rejected as replacement data because they are brand or marketing text, not reviewer emotional expression.

The main completed thesis dataset is the frozen 784-perfume YouTube transcript dataset. It has a clean join between the unified emotion-feature table and Gemini 2.5 Pro reference labels: 784 feature rows, 784 label rows, 784 matched rows, zero duplicate IDs, and zero hash mismatches.

The project attempted a Phase2A expansion toward a 2,000-perfume dataset. That expansion was paused as a documented, resumable branch because YouTube API quota limits, transcript retrieval/IP blocking, low safe-collection yield, and thesis timeline constraints made a clean 2K expansion unrealistic. Phase2A currently has 35 additional usable perfumes and 306 formatted rows, with zero overlap with the frozen 784. It is not merged into the main benchmark at this time. A later merge decision is still open.

The labeling policy was also improved. The original Gemini 2.5 Flash label branch was compared with a Gemini 2.5 Pro judge branch on the same 784 perfumes, same snippets, same schema, same folds, same seeds, and same benchmark protocol. Gemini 2.5 Pro won the branch comparison and received a documented `GO` decision. The Pro branch is now the main model-based reference-label branch for the frozen 784 results.

The final benchmark uses seven official emotion targets: joy, sadness, fear, trust, surprise, anticipation, and disgust. Anger is excluded from the official target set because it has only one positive example in the frozen 784 branch. The main benchmark compares Logistic Regression, Random Forest, XGBoost, BERT, and BiLSTM + Attention. XGBoost has the best macro-F1, while Logistic Regression has the best macro PR-AUC.

NRC and VAD are not separate side notes. They are core feature families used throughout the pipeline. NRC provides transparent lexicon-based discrete emotion counts. VAD provides dimensional affect features: valence, arousal, and dominance. They appear in feature extraction, fusion, supervised modeling, agreement analysis, ablation, feature importance, and discussion.

## Key Dates And Artifacts

Important dates and files to mention:

- `2025-09-20`: several base pipeline scripts were originally created, including preprocessing, lexicon extraction, transformer inference, context extraction, and aggregation scripts.
- `2026-04-13`: top-level thesis workflow and Gemini labeling/training/analysis modules were added or formalized.
- `2026-05-15`: many core scripts and docs were last modified around the frozen benchmark protocol and workflow cleanup.
- `2026-05-24`: Phase 1 Gemini 2.5 Pro judge branch was frozen. The human-readable decision note is `docs/phase1_gemini_pro_judge_decision.md`.
- `2026-06-03`: frozen 784 Gemini 2.5 Pro results summary and Phase2A pause summary were created. Key files are `docs/gemini_g25pro_frozen784_results_summary.md` and `docs/phase2a_collection_pause_summary.md`.
- `2026-06-08`: synthetic review generation work appears in `src/synthetic/generate_synthetic_perfume_reviews.py`; this is in progress and should be treated as a separate experiment, not as completed official benchmark evidence.

Core files:

- Main pipeline runner: `src/pipeline/run_thesis_workflow.py`
- Legacy/base runner: `src/pipeline/run_pipeline.py`
- YouTube scraper: `src/preprocessing/perfume_emotion_scraper.py`
- Phase2A batch collection: `src/preprocessing/run_phase2a_collection_batch.py`
- Transcript formatter: `src/preprocessing/format_transcripts.py`
- NRC/VAD extraction: `src/lexicon/apply_emotion_vad_lexicons.py`
- Transformer scripts: `src/transformers/analyze_emotions_*.py`
- Context extraction: `src/descriptive_terms/extract_context_terms.py`
- Feature fusion: `src/aggregation_fusion/aggregate_and_merge_emotions.py`
- Gemini labeling: `src/labeling/run_gemini_labeling.py`
- Strict Gemini schema: `src/labeling/strict_schema.py`
- Gemini unified analysis: `src/analysis/analyze_gemini_unified.py`
- Flash vs Pro branch comparison: `src/analysis/compare_gemini_branches.py`
- Gold-43 comparison: `src/analysis/compare_label_branch_to_gold43.py`
- Official benchmark config: `config/evaluation_config.yaml`
- Benchmark runners: `src/training/run_*_benchmark.py`
- Stage7 branch-local training: `src/training/train_eval_gemini784.py`
- Deep text dataset builder: `src/training/build_deep_model_text_dataset.py`
- Stage8 feature importance: `src/analysis/analyze_stage8_feature_importance.py`
- Ablation runner: `src/pipeline/ablation_runner.py`
- Result table builder: `scripts/build_result_tables.py`
- Compact result summary builder: `scripts/build_compact_result_summary_image.py`

## Overall Pipeline Summary

The pipeline can be read as a sequence of stages:

1. Collect YouTube perfume review metadata and transcripts.
2. Clean and format transcripts.
3. Extract NRC emotion-count features.
4. Extract NRC VAD valence/arousal/dominance features.
5. Run transformer emotion models on transcript text.
6. Extract descriptive context terms around emotion anchors.
7. Aggregate review-level outputs to perfume-level features.
8. Fuse NRC, VAD, and transformer features into one unified perfume profile.
9. Label perfumes with Gemini using a strict emotion/VAD schema.
10. Compare Gemini Flash and Gemini Pro label branches.
11. Join Gemini labels with the unified feature table.
12. Run agreement analysis between NRC, VAD, transformers, and Gemini labels.
13. Train and evaluate supervised models on the frozen 784 branch.
14. Build deep text inputs for BERT and BiLSTM.
15. Run threshold diagnostics.
16. Run ablation analysis.
17. Run Stage8 feature/family permutation importance.
18. Generate tables, figures, and thesis/advisor-facing summaries.

The main orchestrator, `src/pipeline/run_thesis_workflow.py`, lists the official stage order:

1. `scrape`
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

The `scrape`, `gemini_labeling`, and `evaluation_reports` stages are marked optional/manual in current usage because they involve external APIs, long-running work, or reporting choices. The main completed outputs use frozen artifacts rather than rerunning collection or labeling.

## Data Gathering Design

The original data collection strategy uses YouTube review transcripts. The scraper searches YouTube for perfume-related queries, collects video metadata, fetches transcripts when available, and writes appendable CSV rows.

The main raw files are:

- `data/raw/YouTube_Perfume_Reviews_Partial.csv`
- `data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv`

The raw review-level CSV includes:

- `Perfume`
- `Video ID`
- `Title`
- `Description`
- `URL`
- `Channel Title`
- `Transcript`
- `Transcript Length`
- `Emotion Keyword Count`
- `Final Transcript`

The current main thesis dataset uses the frozen 784-perfume transcript universe. Earlier inventory showed:

- Raw review rows in the frozen file: 4,881
- Raw unique perfumes: 784
- Raw perfumes with usable `Final Transcript`: 784

This matters methodologically because the model is trained and evaluated on reviewer emotional language, not on marketing text.

## Phase2A Expansion Attempt

Phase2A was a controlled plan to expand from 784 perfumes toward 2,000 perfumes using the promoted Gemini 2.5 Pro labeling policy.

Relevant files:

- Plan: `docs/phase2a_2000_perfume_expansion_plan.md`
- Pause summary: `docs/phase2a_collection_pause_summary.md`
- Pause JSON: `outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_collection_pause_summary.json`
- Queue v1: `outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue.csv`
- Queue v2: `outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue_v2.csv`
- Queue v3: `outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue_v3.csv`
- Phase2A raw rows: `data/raw/YouTube_Perfume_Reviews_Phase2A_Partial.csv`
- Phase2A formatted rows: `data/raw/YouTube_Perfume_Reviews_Phase2A_FullyFormatted.csv`
- Phase2A status ledger: `data/intermediate/phase2a_processed_perfumes.csv`

The Phase2A collection branch added:

- 306 raw rows
- 306 formatted rows
- 35 unique usable additional perfumes
- 0 overlap with the frozen 784
- total usable union of 819 perfumes if frozen 784 and Phase2A are counted together

Current Phase2A status counts:

- `done`: 35
- `transcript_fetch_error`: 86
- `transcript_unavailable`: 19
- `api_error`: 5
- `quota_error`: 3
- `search_no_videos`: 1

The key blocker is that API keys help with YouTube Data API search quota, but they do not solve transcript retrieval/IP blocking. Transcript fetching through `youtube_transcript_api` can still fail with IP-related blocking even when search keys are available.

Current decision:

- Phase2A is paused, not deleted.
- Phase2A remains a resumable branch.
- It is not included in the main frozen 784 benchmark.
- A later merge decision is still open.

## Scope Boundaries Discussed

These are the corrected scope decisions:

- Full human ground truth for all 784 perfumes: will not be done.
- Phase2A included in main benchmark: will decide later.
- Synthetic reviews used in official benchmark: in progress.
- Product descriptions and note pyramids as replacement review data: explicitly rejected because they do not provide reviewer emotional language for the pipeline.
- 2,000-perfume labeled benchmark completed: not completed. This refers to a future Phase2A benchmark where 2,000 transcript-supported perfumes would be labeled and benchmarked.

The 2,000-perfume labeled benchmark would mean:

1. collect or identify 2,000 perfumes with usable review/transcript text,
2. label those perfumes under the Gemini 2.5 Pro policy,
3. build a 2K unified feature table,
4. rerun the benchmark on that new dataset.

That has not happened. It remains a future expansion path.

## Transcript Cleaning

The script `src/preprocessing/format_transcripts.py` normalizes transcript formatting. It removes excessive whitespace, replaces line breaks with spaces, and standardizes both `Transcript` and `Final Transcript`.

This step is necessary because all downstream NLP methods assume a stable text field:

- NRC token matching
- NRC VAD token matching
- transformer model input
- context window extraction
- Gemini snippet selection
- deep text benchmark dataset construction

Main output:

- `data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv`

## NRC Emotion Lexicon Features

NRC belongs in the feature-extraction section of the thesis. It is one of the first emotion-analysis methods applied after transcript cleaning.

The script `src/lexicon/apply_emotion_vad_lexicons.py` reads:

- `data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv`
- `data/lexicons/nrc_emotion/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt`

It tokenizes each transcript and counts words associated with NRC categories:

- anger
- anticipation
- disgust
- fear
- joy
- sadness
- surprise
- trust
- positive
- negative

Output:

- `data/perfume_level/NRC_Emotion_Labels_Perfume.csv`

In the thesis, NRC should be described as a transparent lexicon baseline. Its advantage is interpretability. Its limitation is that it does not understand context, negation, irony, perfume-specific use, or whether the emotional word is actually describing the reviewer's tone.

## NRC VAD Features

VAD should be introduced together with NRC, but as a dimensional affect representation.

The same script, `src/lexicon/apply_emotion_vad_lexicons.py`, also reads:

- `data/lexicons/nrc_vad/NRC-VAD-Lexicon-v2.1.txt`

For matched words, it averages:

- valence
- arousal
- dominance

Output:

- `data/perfume_level/VAD_Scores_Perfume.csv`

In the thesis:

- Valence means emotional positivity/negativity.
- Arousal means emotional activation or intensity.
- Dominance means control, power, or assertiveness.

VAD appears again later as:

- input features in the unified table,
- comparison targets against Gemini VAD,
- secondary supervised regression targets,
- ablation families,
- Stage8 feature-importance families.

## Transformer Emotion Feature Extraction

The pipeline uses four transformer emotion classifiers:

- `src/transformers/analyze_emotions_distilroberta_jhartmann.py`
  - model: `j-hartmann/emotion-english-distilroberta-base`
  - output: `data/intermediate/YouTube_Perfume_Reviews_With_Transformer.csv`
- `src/transformers/analyze_emotions_distilbert_savani.py`
  - model: `bhadresh-savani/distilbert-base-uncased-emotion`
  - output: `data/intermediate/YouTube_Perfume_Reviews_With_Savani.csv`
- `src/transformers/analyze_emotions_goemotions_multi.py`
  - model: `monologg/bert-base-cased-goemotions-original`
  - output: `data/intermediate/YouTube_Perfume_Reviews_With_GoEmotions.csv`
- `src/transformers/analyze_emotions_roberta_cardiffnlp.py`
  - model: `cardiffnlp/twitter-roberta-base-emotion`
  - output: `data/intermediate/YouTube_Perfume_Reviews_With_CardiffRoberta.csv`

The transformer outputs provide contextual emotion probabilities. The thesis should contrast these with NRC: transformers are less transparent but more context-aware.

## Context Terms And Descriptive Terms

The script `src/descriptive_terms/extract_context_terms.py` extracts context words around emotion anchors. It uses a small token window around emotion terms and builds per-perfume context keyword counts.

Output:

- `data/perfume_level/perfume_context_keywords.csv`

The script `src/descriptive_terms/analyze_top_terms.py` filters terms, prints global and per-perfume top terms, and optionally saves intensifier counts.

Output:

- `data/perfume_level/perfume_intensifier_counts.csv`

This part supports interpretability. It helps explain what descriptive vocabulary appears around emotional language in perfume reviews.

## Optional Scent-Emotion Reprocessing

The script `src/lexicon/reprocess_scent_emotion.py` uses expanded scent and emotion lexicons to tag transcripts with scent terms and emotion terms.

Output:

- `data/intermediate/YouTube_Perfume_Reviews_With_ScentEmotion.csv`

This is an optional enrichment stage. It should be described as qualitative or exploratory support, not as the main benchmark target.

## Aggregation And Fusion

Feature aggregation happens in:

- `src/aggregation_fusion/merge_emotion_profiles.py`
- `src/aggregation_fusion/aggregate_and_merge_emotions.py`

The stronger final fusion script is `aggregate_and_merge_emotions.py`. It builds a perfume-level table by:

- summing NRC emotion counts per perfume,
- averaging VAD values per perfume,
- expanding Hartmann probability dictionaries,
- averaging numeric transformer score columns per perfume,
- joining categorical top-label modes per perfume.

Main output:

- `data/perfume_level/Unified_Emotion_Profile_Perfume.csv`

This file is central to the thesis. It is the unified feature representation used for agreement analysis, Gemini joining, supervised training, ablation, and feature importance.

The feature families are:

- `NRC_`
- `VAD_`
- `Hartmann_`
- `Savani_`
- `GoEmotions_`
- `Cardiff_`

## Silver Labels And Fused Labels

The repo includes early weak-supervision/silver-label scripts:

- `src/labeling/build_silver_labels.py`
- `src/labeling/create_silver_labels_fused.py`

The first creates silver labels from majority vote over transformer top labels. The second uses score-based fusion by projecting transformer scores into a shared coarse label space and combining them.

Outputs:

- `labels/silver/Unified_With_Silver_Labels.csv`
- `labels/silver/Unified_With_Silver_FusedLabels.csv`

How to describe this:

- Silver labels were exploratory weak labels.
- They helped test fusion ideas.
- They are not the final reference labels.
- The final reference-label branch uses Gemini 2.5 Pro.

## Gemini Labeling Protocol

The Gemini labeling pipeline is implemented in:

- `src/labeling/run_gemini_labeling.py`
- `src/labeling/gemini_client.py`
- `src/labeling/strict_schema.py`
- `src/labeling/aggregate_snippets.py`
- `src/labeling/reporting.py`

The strict schema includes:

- binary emotions: joy, sadness, anger, fear, trust, surprise, anticipation, disgust
- VAD dimensions: valence, arousal, dominance

Each perfume is represented by three snippets. Each snippet is labeled independently. The perfume-level aggregation rule is:

- emotion labels: majority vote,
- VAD scores: mean.

Main Gemini Pro branch label files:

- `labels/gemini_labels_784_judge_g25pro_v1.csv`
- `labels/gemini_labels_784_judge_g25pro_v1.jsonl`
- `labels/gemini_labeling_report_judge_g25pro_v1.json`
- `labels/gemini_labeling_errors_judge_g25pro_v1.jsonl`
- `labels/gemini_labeling_run_judge_g25pro_v1.log`

Important wording:

Gemini labels are improved model-based reference labels. They are not perfect human ground truth.

## Gemini Flash Baseline Vs Gemini 2.5 Pro Branch

This is one of the major methodological decisions.

The baseline label file was:

- `labels/gemini_labels_784_v1.csv`

The Pro branch label file was:

- `labels/gemini_labels_784_judge_g25pro_v1.csv`

The comparison was controlled: same 784 perfumes, same review universe, same snippets, same schema, same aggregation rule, same folds, same seeds, same benchmark protocol. The intended variable was the judge model.

Key files:

- `config/gemini_judge/judge_g25pro_v1.yaml`
- `src/pipeline/run_gemini_judge_branch.py`
- `src/analysis/compare_label_branch_to_gold43.py`
- `src/analysis/compare_gemini_branches.py`
- `src/analysis/freeze_phase1_gemini_pro_branch.py`
- `docs/phase1_gemini_pro_judge_decision.md`
- `outputs/gemini_judge/judge_g25pro_v1/decision/branch_decision_summary.json`
- `outputs/gemini_judge/judge_g25pro_v1/phase1_freeze_manifest.json`

Decision result: `GO`.

Key metrics:

- Flash gold macro-F1: 0.268197
- Gemini 2.5 Pro gold macro-F1: 0.432900
- Flash benchmark mean macro-F1: 0.238986
- Gemini 2.5 Pro benchmark mean macro-F1: 0.283992

Interpretation:

Gemini 2.5 Pro improved both the small gold sanity-check metric and downstream benchmark mean macro-F1. Therefore, the Pro branch was promoted as the stronger labeling policy for the main frozen 784 thesis results.

## Gold-43 Sanity Check

The file `data/annotation_packets/Labeled_Annotation_Packet_strict_v1.csv` is the strict 43-row gold annotation packet.

It is used for secondary sanity checking and branch comparison. It is not full human ground truth for all 784 perfumes.

Current scope decision:

- full human labels for all 784 perfumes will not be created.
- Gemini 2.5 Pro labels remain the model-based reference labels.
- the 43-row packet is useful as a small sanity check.

## Gemini-Feature Join And Integrity

The script `src/analysis/analyze_gemini_unified.py` joins Gemini labels with the unified feature table.

Main frozen Pro output:

- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/gemini_unified_merged.csv`

Integrity results:

- unified feature rows: 784
- Gemini label rows: 784
- matched rows: 784
- unified-only rows: 0
- Gemini-only rows: 0
- duplicate unified IDs: 0
- duplicate Gemini IDs: 0
- hash mismatches: 0

This is one of the cleanest arguments for the thesis design: the main dataset is internally consistent and does not mix Phase2A.

## Gemini Label Distribution

The frozen Gemini 2.5 Pro branch is highly imbalanced:

- joy: 503 positives, 64.16 percent
- anticipation: 308 positives, 39.29 percent
- trust: 236 positives, 30.10 percent
- sadness: 12 positives, 1.53 percent
- surprise: 10 positives, 1.28 percent
- disgust: 8 positives, 1.02 percent
- fear: 3 positives, 0.38 percent
- anger: 1 positive, 0.13 percent

Interpretation:

Common perfume emotions such as joy, anticipation, and trust are more learnable. Rare emotions are unstable and should be interpreted cautiously. Anger is excluded from the official benchmark because it has only one positive example.

Gemini VAD target means:

- valence: 6.679
- arousal: 5.950
- dominance: 5.916

This shows that the corpus is generally positive and moderately activated.

## Agreement Analysis

Agreement outputs are under:

- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/agreement_binary_nrc_vs_gemini.csv`
- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/agreement_toplabel_models_vs_gemini.csv`
- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/agreement_vad_continuous.csv`

NRC vs Gemini:

- Mean binary accuracy: 0.184

This is low because the NRC rule is broad: if any word linked to an emotion appears, that emotion is marked present. This creates many lexical positives and does not capture contextual perfume-level emotional tone.

Transformer top-label agreement with Gemini:

- Cardiff: 0.635
- Savani: 0.631
- GoEmotions: 0.588
- Hartmann: 0.506

VAD agreement between lexicon VAD and Gemini VAD:

- valence Pearson r: 0.252
- arousal Pearson r: 0.218
- dominance Pearson r: 0.268

Interpretation:

Transformer models align better with Gemini's contextual labels than raw NRC binary matching. Lexicon VAD has modest positive directional alignment with Gemini VAD, but the scales should not be treated as interchangeable.

## Stage7 Branch-Local Training

The script `src/training/train_eval_gemini784.py` trains branch-local models on:

- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/gemini_unified_merged.csv`

It uses 58 numeric features from:

- NRC
- VAD
- Hartmann
- Savani
- GoEmotions
- Cardiff

Outputs:

- `outputs/gemini_judge/judge_g25pro_v1/training/stage7_gemini784_metrics.json`
- `outputs/gemini_judge/judge_g25pro_v1/training/stage7_gemini784_oof_predictions.csv`
- `outputs/gemini_judge/judge_g25pro_v1/training/stage7_gemini784_label_diagnostics.csv`
- `outputs/gemini_judge/judge_g25pro_v1/training/stage7_gemini784_vad_metrics.csv`

Stage7 global emotion metrics:

- macro-F1 over nondegenerate labels: 0.300918
- micro-F1 over nondegenerate labels: 0.513817
- subset accuracy: 0.200255
- hamming loss: 0.179528
- label cardinality: 1.378827

Stage7 VAD average metrics:

- MAE: 0.635440
- RMSE: 0.826219
- R2: 0.040846
- Pearson r: 0.322877
- Spearman r: 0.305091

Interpretation:

VAD prediction is possible but modest. Valence is strongest, arousal is moderate, and dominance is weakest.

## Official Benchmark Design

The official benchmark is defined in:

- `config/evaluation_config.yaml`
- `docs/evaluation_protocol.md`
- `docs/artifact_contract.md`

Canonical input:

- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/gemini_unified_merged.csv`

Official target labels:

- joy
- sadness
- fear
- trust
- surprise
- anticipation
- disgust

Excluded label:

- anger, because it has only one positive example

Frozen split design:

- 5-fold KFold
- shuffle: true
- random state: 42
- unit: perfume row / `perfume_id`
- fold manifest: `outputs/gemini_judge/judge_g25pro_v1/model_benchmark/tables/fold_assignments.csv`

Official threshold:

- 0.3

Main metrics:

- macro-F1
- macro ROC-AUC
- macro PR-AUC

Models:

- Logistic Regression
- Random Forest
- XGBoost
- BERT
- BiLSTM + Attention

## Benchmark Results

Final table:

- `outputs/gemini_judge/judge_g25pro_v1/model_benchmark/tables/final_model_comparison_for_thesis.csv`

Results:

| Model | Macro-F1 | Macro ROC-AUC | Macro PR-AUC |
|---|---:|---:|---:|
| XGBoost | 0.318425 | 0.741066 | 0.303108 |
| Logistic Regression | 0.317810 | 0.729397 | 0.326201 |
| BiLSTM + Attention | 0.269639 | 0.487741 | 0.256811 |
| BERT | 0.258405 | 0.542510 | 0.218617 |
| Random Forest | 0.255681 | 0.728696 | 0.296820 |

Interpretation:

XGBoost has the best macro-F1. Logistic Regression is almost tied and has the best macro PR-AUC. Classical tabular models perform better than the neural text models in this small, imbalanced benchmark. This is defensible because the dataset is small, labels are imbalanced, and the tabular models directly use engineered emotion features.

Strongest labels across benchmark models:

- joy
- anticipation
- trust

Weakest labels:

- fear
- disgust
- surprise
- sadness

The weak labels are rare. Their low F1 scores are a data support issue, not simply a model failure.

## Deep Text Benchmark Dataset

Deep text inputs are built by:

- `src/training/build_deep_model_text_dataset.py`

Outputs:

- `outputs/gemini_judge/judge_g25pro_v1/model_benchmark/datasets/perfume_text_dataset_v1.jsonl`
- `outputs/gemini_judge/judge_g25pro_v1/model_benchmark/datasets/perfume_text_dataset_v1_chunks.csv`
- `outputs/gemini_judge/judge_g25pro_v1/model_benchmark/datasets/perfume_text_dataset_v1_summary.json`

The builder creates one perfume-level JSON object per perfume, plus chunk-level audit rows. Chunks use title, description, and transcript excerpts from real review rows. Labels and fold IDs are attached so BERT and BiLSTM use the same evaluation protocol as the classical benchmark.

## Threshold Tuning And Threshold Sweep

The official benchmark uses threshold 0.3. The repo also includes threshold tuning and threshold sweep diagnostics.

Files:

- `src/training/threshold_tuning.py`
- `src/training/run_fixed_threshold_sweep.py`
- `src/training/generate_threshold_sweep_summary.py`
- `src/training/generate_available_threshold_comparison_table.py`

Outputs include:

- `outputs/model_benchmark_threshold_sweep/tables/final_model_threshold_comparison_metric_grouped_03_05_07.csv`
- `outputs/model_benchmark_threshold_sweep/figures/final_model_threshold_comparison_metric_grouped_03_05_07.png`
- `outputs/model_benchmark_threshold_sweep/figures/macro_f1_vs_threshold_by_model.png`

Interpretation:

Threshold analysis is a diagnostic for imbalanced multilabel classification. It helps show that threshold choice matters, but it does not replace the official fixed 0.3 benchmark row.

## Ablation Study

The ablation study is implemented in:

- `src/pipeline/ablation_runner.py`

Main output:

- `outputs/ablation_results/summary.csv`

The ablation compares feature variants against the full hybrid feature representation:

- `FULL_HYBRID`
- `LEXICON_ONLY`
- `TRANSFORMER_ONLY`
- `NO_NRC`
- `NO_VAD`
- `HARTMANN_ONLY`
- `SAVANI_ONLY`
- `GOEMOTIONS_ONLY`
- `CARDIFF_ONLY`
- `LABELS_ONLY`

Metrics:

- Spearman correlation of pairwise cosine distances vs full hybrid
- nearest-neighbor overlap vs full hybrid
- KMeans adjusted Rand index vs full hybrid
- ranking stability on selected axes

Important results:

- `TRANSFORMER_ONLY`: Spearman 0.929, kNN overlap 0.753
- `LEXICON_ONLY`: Spearman 0.344, kNN overlap 0.066
- `NO_NRC`: Spearman 0.971, kNN overlap 0.892
- `NO_VAD`: Spearman 0.960, kNN overlap 0.811
- `GOEMOTIONS_ONLY`: Spearman 0.662, kNN overlap 0.388

Interpretation:

Transformer features preserve the full hybrid geometry much better than lexicon-only features. Removing NRC or VAD still leaves high geometry preservation because the transformer families dominate the full representation. However, NRC and VAD remain useful for interpretability and VAD-related analysis.

## Stage8 Feature Importance

Stage8 is implemented in:

- `src/analysis/analyze_stage8_feature_importance.py`

Outputs:

- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/stage8_emotion_feature_importance.csv`
- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/stage8_emotion_family_importance.csv`
- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/stage8_vad_feature_importance.csv`
- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/stage8_vad_family_importance.csv`
- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/stage8_importance_summary_report.json`

Stage8 computes validation-safe permutation importance on held-out folds.

Discrete emotion feature-family importance:

- Hartmann: 0.077896
- GoEmotions: 0.044274
- Cardiff: 0.036566
- Savani: 0.013676
- VAD: 0.002811
- NRC: 0.000000

VAD feature-family importance:

- GoEmotions: 0.068603
- Cardiff: 0.024766
- VAD: 0.014443
- NRC: 0.007529
- Hartmann: 0.004544
- Savani: 0.001761

Interpretation:

Transformer-derived features dominate discrete Gemini emotion prediction. NRC is weak as a direct binary label match but can still contribute to VAD prediction. VAD is useful as a compact dimensional feature family, but it is not the strongest family for discrete labels.

## Synthetic Reviews

Synthetic reviews are now a separate pilot experiment rather than official frozen-784 benchmark evidence.

Current memory file:

- `docs/synthetic_review_experiment_memory.md`

Current v2 pilot:

- generated by `src/synthetic/generate_emotion_rich_synthetic_perfume_reviews.py`
- `data/synthetic/synthetic_perfume_reviews_784_emotion_rich_v2_pilot100.csv`
- 100 rows, 20 perfumes, 5 reviews per perfume, seed `20260612`
- no emotion/VAD labels generated at text-generation time
- uses local note-inspired descriptors from `data/perfume_level/perfume_context_keywords.csv`

Review-level pilot labels:

- `labels/gemini_review_level/gemini_review_labels_emotion_rich_v2_pilot100_g25pro_v1_pipeline.csv`
- Gemini 2.5 Pro labels were used only for this pilot comparison because the original OG Pro branch also used Gemini 2.5 Pro.

100-vs-100 comparison:

- synthetic: `outputs/synthetic_review_level_tests/emotion_rich_v2_pilot100/supervised_eval_g25pro/logistic_regression_review_level_report.md`
- YouTube: `outputs/youtube_review_level_tests/pilot100_g25pro/supervised_eval_g25pro/logistic_regression_review_level_report.md`
- comparison: `outputs/review_level_pilot100_comparison_g25pro/pilot100_comparison_report.md`

Key pilot result:

- synthetic v2 100 macro-F1: 0.6503
- YouTube 100 all-evaluable macro-F1: 0.4514
- YouTube 100 common-six macro-F1: 0.4954

Interpretation:

- Synthetic v2 is easier and cleaner for the LR pipeline to learn, but this does not prove it is more realistic or better thesis data.
- The full synthetic v2 corpus should not be merged into the frozen 784 benchmark unless it is explicitly framed as a separate augmentation experiment.
- The main thesis benchmark remains the frozen 784 perfume-level YouTube dataset.

## Evaluation Reports And Figures

Evaluation and visualization use:

- `src/evaluation/evaluate_models.py`
- `src/training/generate_final_benchmark_figures.py`
- `scripts/build_result_tables.py`
- `scripts/build_compact_result_summary_image.py`

Outputs include:

- `outputs/evaluation_out/pairwise_kappa.csv`
- `outputs/evaluation_out/fleiss_kappa.txt`
- `outputs/evaluation_out/agreement_by_perfume.csv`
- `outputs/evaluation_out/label_distribution_by_model.csv`
- `outputs/evaluation_out/jsd_pairwise.csv`
- `outputs/evaluation_out/vad_correlations.csv`
- `outputs/evaluation_out/vad_color_palette_hsv.csv`
- `outputs/figures/kappa_heatmap.png`
- `outputs/figures/jsd_heatmap.png`
- `outputs/figures/nrc_heatmap.png`
- `outputs/figures/vad_color_swatches.png`
- `docs/flash_vs_pro_result_tables/`

These files are useful for thesis figures and advisor presentation slides.

## Main Methodological Choices

### Why YouTube transcripts?

The thesis needs emotion-bearing review language. YouTube reviews contain subjective reactions, affective descriptions, and personal responses. Product descriptions and note pyramids do not serve the same purpose because they are usually brand-authored and marketing-oriented.

### Why freeze the 784 dataset?

The 784 dataset has complete transcript coverage, complete Gemini Pro labels, complete feature joins, and complete benchmark outputs. It is methodologically cleaner to proceed with a complete frozen dataset than to force a low-yield expansion with inconsistent data quality.

### Why pause Phase2A?

Phase2A was paused because the data collection bottleneck was not just search quota. Transcript fetching itself became blocked or unreliable. Continuing under those conditions would risk creating a biased or low-quality expansion.

### Why Gemini 2.5 Pro?

Gemini 2.5 Pro beat Gemini 2.5 Flash in the controlled branch comparison. It improved both the 43-row gold sanity-check macro-F1 and downstream benchmark mean macro-F1.

### Why not call Gemini labels perfect ground truth?

They are model-generated reference labels. They are useful, consistent, and improved relative to the Flash baseline, but they are not human consensus labels.

### Why include NRC and VAD?

NRC and VAD provide transparent, interpretable affect features. They are weaker than contextual transformer features for direct Gemini emotion prediction, but they support interpretability, dimensional affect analysis, ablation, and feature-importance discussion.

### Why do classical models perform well?

The dataset is small and imbalanced. Classical tabular models use engineered emotion features directly. Neural text models have less data to learn from and are more sensitive to rare-label imbalance.

## Limitations

- Full human ground truth for all 784 perfumes is outside scope.
- Gemini labels are model-based references, not perfect ground truth.
- The label distribution is highly imbalanced.
- Rare labels such as fear, disgust, surprise, sadness, and anger are unstable.
- Anger is excluded from the official benchmark because it has only one positive example.
- NRC matching is transparent but context-poor.
- VAD sources are not directly interchangeable because lexicon VAD and Gemini VAD operate on different practical scales.
- Phase2A is not included in the main benchmark.
- Synthetic reviews are in progress and should not yet be framed as completed official benchmark evidence.

## Suggested Thesis Section Placement

Methods:

- Data source and transcript collection
- Transcript preprocessing
- NRC emotion lexicon extraction
- NRC VAD extraction
- Transformer emotion inference
- Feature fusion
- Gemini labeling protocol
- Benchmark protocol
- Ablation and feature importance

Results:

- Dataset integrity
- Gemini Flash vs Pro branch decision
- Label distribution and imbalance
- NRC/transformer/VAD agreement analysis
- Benchmark model comparison
- VAD prediction results
- Ablation results
- Stage8 feature importance

Discussion:

- Why frozen 784 is methodologically cleaner
- Why Phase2A was paused
- Why rare labels are limited
- Why NRC is interpretable but weak for contextual labels
- Why transformer features dominate
- Why Gemini Pro is a reference label, not human ground truth
- Future work: Phase2A merge decision and synthetic review experiment

## Suggested Advisor Talking Points

1. The main thesis dataset is complete and clean: frozen 784 perfumes, 784 labels, 784 matched feature rows.
2. Gemini 2.5 Pro replaced Gemini Flash because it performed better in a controlled branch comparison.
3. The 2K expansion was attempted seriously, but paused because transcript collection was blocked by platform constraints and low yield.
4. NRC and VAD are included as interpretable affect feature families, but transformer features carry stronger contextual signal.
5. XGBoost and Logistic Regression are the strongest benchmark models because the dataset is small, tabular, and imbalanced.
6. Rare emotions are not reliable enough for strong claims; joy, anticipation, and trust are the stable labels.
7. Phase2A remains separate and can be revisited later.
8. Synthetic reviews are in progress as a separate experiment, not yet part of completed official results.

## One-Paragraph Thesis Summary

This thesis develops a hybrid NLP pipeline for perfume emotion analysis using YouTube review transcripts. The pipeline extracts transparent lexicon-based NRC emotion counts, dimensional NRC VAD scores, and contextual transformer emotion features, then fuses them into a perfume-level emotional profile. A controlled label-branch comparison promoted Gemini 2.5 Pro over Gemini 2.5 Flash as the model-based reference-label policy. The final frozen 784-perfume dataset has complete feature-label alignment and is used for agreement analysis, supervised benchmarking, ablation, and feature-importance interpretation. The attempted 2,000-perfume Phase2A expansion was paused as a resumable branch because transcript collection was limited by quota and IP blocking. The main results therefore proceed on the clean frozen 784 dataset, while Phase2A and synthetic review generation remain future or in-progress extensions.
