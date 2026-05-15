<!--
Date Created: 2026-04-13 23:02:44+03:00
Date Last Modified: 2026-05-15
-->

# Comparison Table Template

## Official columns

Use the CSV version of this template as the source of truth for the main table.

Completion rule:

- `row_complete = true` only if `reproducible_run`, `saved_metrics`, `comparable_plots`, and `interpretation_note` are all `true`

## Markdown view

| model_id | model_family | dataset | granularity | target_columns | split_design | data_seed | model_seed | threshold | macro_f1 | macro_roc_auc | macro_pr_auc | reproducible_run | saved_metrics | comparable_plots | interpretation_note | row_complete | note_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | Logistic Regression | outputs/analysis_out/gemini_unified_merged.csv | perfume_level | joy;sadness;fear;trust;surprise;anticipation;disgust | 5-fold KFold | 42 | 42 | 0.3 |  |  |  | false | false | false | false | false | outputs/model_benchmark/tables/logistic_regression_note.md |
| xgboost | XGBoost | outputs/analysis_out/gemini_unified_merged.csv | perfume_level | joy;sadness;fear;trust;surprise;anticipation;disgust | 5-fold KFold | 42 | 42 | 0.3 |  |  |  | false | false | false | false | false | outputs/model_benchmark/tables/xgboost_note.md |
| random_forest | Random Forest | outputs/analysis_out/gemini_unified_merged.csv | perfume_level | joy;sadness;fear;trust;surprise;anticipation;disgust | 5-fold KFold | 42 | 42 | 0.3 |  |  |  | false | false | false | false | false | outputs/model_benchmark/tables/random_forest_note.md |
| bert | BERT | outputs/analysis_out/gemini_unified_merged.csv | perfume_level | joy;sadness;fear;trust;surprise;anticipation;disgust | 5-fold KFold | 42 | 42 | 0.3 |  |  |  | false | false | false | false | false | outputs/model_benchmark/tables/bert_note.md |
| bilstm_attention | BiLSTM + Attention | outputs/analysis_out/gemini_unified_merged.csv | perfume_level | joy;sadness;fear;trust;surprise;anticipation;disgust | 5-fold KFold | 42 | 42 | 0.3 |  |  |  | false | false | false | false | false | outputs/model_benchmark/tables/bilstm_attention_note.md |
