<!--
Date Created: 2026-06-03
-->

# Phase2A Collection Pause Summary

## Decision

Phase2A YouTube transcript expansion is paused as a documented, resumable collection branch. It is not deleted, abandoned, or merged into the primary thesis dataset at this point.

The thesis should now proceed on the frozen 784-perfume dataset as the main dataset. The Phase2A supplement stays separate unless a later, explicit merge decision is made after substantially more transcript perfumes are collected and labeled under the same protocol.

## Current State

| Item | Current value |
|---|---:|
| Frozen Phase1 / judge dataset perfumes | 784 |
| Phase2A raw rows | 306 |
| Phase2A formatted rows | 306 |
| Unique Phase2A perfumes | 35 |
| Phase2A overlap with frozen 784 | 0 |
| Total usable including frozen 784 | 819 |
| Phase2A processed status rows | 149 |

Phase2A processed status counts:

| Status | Count |
|---|---:|
| `done` | 35 |
| `transcript_fetch_error` | 86 |
| `transcript_unavailable` | 19 |
| `api_error` | 5 |
| `quota_error` | 3 |
| `search_no_videos` | 1 |

Source artifacts:

- Raw Phase2A rows: `data/raw/YouTube_Perfume_Reviews_Phase2A_Partial.csv`
- Formatted Phase2A rows: `data/raw/YouTube_Perfume_Reviews_Phase2A_FullyFormatted.csv`
- Phase2A status ledger: `data/intermediate/phase2a_processed_perfumes.csv`
- Frozen judge labels: `labels/gemini_labels_784_judge_g25pro_v1.csv`
- Frozen Phase1 manifest: `outputs/gemini_judge/judge_g25pro_v1/phase1_freeze_manifest.json`

## Queue State

Queue versions present:

| Version | Path | Rows |
|---|---|---:|
| v1 | `outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue.csv` | 50 |
| v2 | `outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue_v2.csv` | 2191 |
| v3 | `outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue_v3.csv` | 2191 |

Current recommended queue path:

```bash
outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue_v3.csv
```

Current safe collection mode:

```bash
--old-pace
```

Regenerate queue v4 only if queue v3 becomes exhausted or polluted.

## Blockers

- YouTube Data API quota limits.
- `youtube_transcript_api` `transcript_fetch_error` / `IpBlocked`.
- Low yield under safe collection settings.
- Limited thesis timeline.

Under these constraints, reaching 2,000 usable YouTube transcript perfumes is not realistic for the current thesis schedule.

## How To Resume Phase2A Later

Resume only after quota and network conditions improve.

1. Wait for YouTube quota reset or obtain a quota increase.
2. Use a non-blocked network for transcript retrieval.
3. Run the transcript diagnostic:

```bash
python3 src/preprocessing/run_phase2a_collection_batch.py --diagnose-transcript-error YXcy3En2gCA
```

4. Diagnose YouTube API keys:

```bash
python3 src/preprocessing/perfume_emotion_scraper.py --diagnose-youtube-keys --youtube-key-file .env.youtube_keys --max-keys 13
```

5. Run old-pace collection:

```bash
python3 src/preprocessing/run_phase2a_collection_batch.py --old-pace
```

6. Count the results:

- Raw rows.
- Formatted rows.
- Unique Phase2A perfumes.
- Processed status counts.
- Total usable union with the frozen 784.

7. Regenerate queue v4 only if queue v3 becomes exhausted or polluted.

## What Not To Do When Resuming

- Do not run large 13-key batches immediately.
- Do not retry `transcript_fetch_error` unless the IP diagnostic passes and retrying is explicitly intended.
- Do not mix Description+Notes as a replacement for YouTube review transcripts.
- Do not overwrite frozen Phase1 artifacts.
- Do not commit `.env` or `.env.youtube_keys`.

## Scoring And Evaluation Pipeline

The current scoring/evaluation pipeline already supports the frozen 784 branch.

| Component | Script/config | Main artifacts |
|---|---|---|
| NRC emotion extraction | `src/lexicon/apply_emotion_vad_lexicons.py` | `data/perfume_level/NRC_Emotion_Labels_Perfume.csv` |
| VAD mapping | `src/lexicon/apply_emotion_vad_lexicons.py` | `data/perfume_level/VAD_Scores_Perfume.csv` |
| Aggregation/fusion | `src/aggregation_fusion/aggregate_and_merge_emotions.py` | `data/perfume_level/Unified_Emotion_Profile_Perfume.csv` |
| Gemini Pro labels | `labels/gemini_labels_784_judge_g25pro_v1.csv` | `outputs/gemini_judge/judge_g25pro_v1/gold_eval/judge_vs_gold_metrics.json` |
| Gemini-feature join | `src/analysis/analyze_gemini_unified.py` | `outputs/gemini_judge/judge_g25pro_v1/analysis_out/gemini_unified_merged.csv` |
| Official benchmark | `config/evaluation_config.yaml` and `src/training/run_*_benchmark.py` | `outputs/gemini_judge/judge_g25pro_v1/model_benchmark/` |
| Visual mapping/evaluation | `src/evaluation/evaluate_models.py` | `outputs/evaluation_out/`, `outputs/figures/` |
| Feature importance | `src/analysis/analyze_stage8_feature_importance.py` | `outputs/analysis_out/stage8_*` |

Current frozen judge benchmark outputs are complete for five models at threshold `0.3`:

| Model | Macro-F1 | Macro ROC-AUC | Macro PR-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.317810 | 0.729397 | 0.326201 |
| XGBoost | 0.318425 | 0.741066 | 0.303108 |
| Random Forest | 0.255681 | 0.728696 | 0.296820 |
| BERT | 0.258405 | 0.542510 | 0.218617 |
| BiLSTM + Attention | 0.269639 | 0.487741 | 0.256811 |

## Next Executable Scoring Plan

Use the frozen 784 artifacts, not Phase2A, for the main thesis results.

Inputs:

- `data/perfume_level/Unified_Emotion_Profile_Perfume.csv`
- `labels/gemini_labels_784_judge_g25pro_v1.csv`
- `data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv`
- `config/evaluation_config.yaml`
- `outputs/model_benchmark/tables/fold_assignments.csv`

Recommended commands:

```bash
python3 -m src.analysis.analyze_gemini_unified --unified-csv data/perfume_level/Unified_Emotion_Profile_Perfume.csv --gemini-csv labels/gemini_labels_784_judge_g25pro_v1.csv --out-dir outputs/gemini_judge/judge_g25pro_v1/analysis_out --perfume-col Perfume --strict
```

```bash
python3 -m src.training.generate_final_benchmark_figures --source-table outputs/gemini_judge/judge_g25pro_v1/model_benchmark/tables/comparison_table.csv --output-dir outputs/gemini_judge/judge_g25pro_v1/model_benchmark
```

```bash
python3 -m src.evaluation.evaluate_models --topN 15 --topK 15 --normalize_vad
```

```bash
python3 -m src.analysis.analyze_stage8_feature_importance --merged-csv outputs/analysis_out/gemini_unified_merged.csv --stage7-metrics-json outputs/training/stage7_gemini784_metrics.json --stage7-label-diagnostics-csv outputs/training/stage7_gemini784_label_diagnostics.csv --stage7-oof-csv outputs/training/stage7_gemini784_oof_predictions.csv --out-dir outputs/analysis_out --n-repeats 5 --strict
```

Primary outputs:

- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/gemini_unified_merged.csv`
- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/gemini_unified_summary_report.json`
- `outputs/gemini_judge/judge_g25pro_v1/gold_eval/judge_vs_gold_metrics.json`
- `outputs/gemini_judge/judge_g25pro_v1/model_benchmark/tables/comparison_table.csv`
- `outputs/gemini_judge/judge_g25pro_v1/model_benchmark/tables/final_model_comparison_for_thesis.csv`
- `outputs/gemini_judge/judge_g25pro_v1/model_benchmark/figures/all_models_grouped_metric_comparison.png`
- `outputs/evaluation_out/pairwise_kappa.csv`
- `outputs/evaluation_out/jsd_pairwise.csv`
- `outputs/evaluation_out/vad_correlations.csv`
- `outputs/evaluation_out/top_disagreements_top15.csv`
- `outputs/evaluation_out/vad_color_palette_hsv.csv`
- `outputs/analysis_out/stage8_emotion_feature_importance.csv`
- `outputs/analysis_out/stage8_emotion_family_importance.csv`
- `outputs/analysis_out/stage8_vad_feature_importance.csv`
- `outputs/analysis_out/stage8_vad_family_importance.csv`

Metrics, tables, and figures produced:

- Gemini-feature join integrity and agreement tables.
- Gold-43 Gemini sanity-check metrics.
- Five-model benchmark table with macro-F1, macro ROC-AUC, and macro PR-AUC.
- Per-label metrics and ROC/PR curves for benchmark models.
- Final model comparison table and thesis-ready model comparison figures.
- Pairwise kappa, Fleiss kappa, JSD, VAD correlations, top disagreements, NRC heatmaps, and VAD color palette.
- Feature-level and family-level importance tables for emotion and VAD interpretation.

## Phase2A Supplement Recommendation

Keep the 35-perfume Phase2A supplement separate for now. It can be used as supplementary qualitative evidence or examples, but it should not be merged into the main quantitative benchmark until there are enough additional transcript perfumes to justify a new labeled dataset version.

Suggested thesis wording:

> The originally planned YouTube transcript expansion to 2,000 perfumes was paused because API quota limits, transcript retrieval IP blocking, low collection yield, and thesis timeline constraints made full expansion infeasible. The expansion branch was preserved as a resumable Phase2A supplement, while the main scoring and evaluation experiments use the frozen 784-perfume transcript dataset with Gemini Pro labels.
