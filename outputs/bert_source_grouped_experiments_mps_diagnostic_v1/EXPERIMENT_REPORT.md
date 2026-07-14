# Source-Grouped BERT Experiments

Status: **partial** (1/14 conditions complete).

All held-out evaluation uses the 784 original YouTube perfumes and the repaired source-component folds.
Synthetic reviews for outer-test and inner-validation perfume IDs are excluded from training.

| Stage | Condition | Synthetic reviews | Macro-F1 | Macro ROC-AUC | Macro PR-AUC | Wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| review_count | weighted_synthetic_reviews_0 | 0 | 0.3253 | 0.5875 | 0.3352 | 40m 44s |

SMOTE: **not applicable**. Not applicable to raw-text BERT: SMOTE interpolates continuous feature vectors, whereas token-ID sequences are discrete and interpolated embeddings would not correspond to observed review texts.
