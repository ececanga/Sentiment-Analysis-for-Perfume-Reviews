# Source-Grouped BERT Experiments

Status: **partial** (6/14 conditions complete).

All held-out evaluation uses the 784 original YouTube perfumes and the repaired source-component folds.
Synthetic reviews for outer-test and inner-validation perfume IDs are excluded from training.

| Stage | Condition | Synthetic reviews | Macro-F1 | Macro ROC-AUC | Macro PR-AUC | Wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| review_count | weighted_synthetic_reviews_0 | 0 | 0.4029 | 0.5533 | 0.3590 | 1h 13m |
| review_count | weighted_synthetic_reviews_99 | 99 | 0.3311 | 0.6166 | 0.3451 | 1h 09m |
| review_count | weighted_synthetic_reviews_201 | 201 | 0.3377 | 0.5879 | 0.3589 | 1h 05m |
| review_count | weighted_synthetic_reviews_300 | 300 | 0.3424 | 0.5918 | 0.3344 | 1h 14m |
| review_count | weighted_synthetic_reviews_501 | 501 | 0.3315 | 0.5407 | 0.3442 | 1h 09m |
| review_count | weighted_synthetic_reviews_1002 | 1002 | 0.3317 | 0.5414 | 0.3168 | 1h 12m |

SMOTE: **not applicable**. Not applicable to raw-text BERT: SMOTE interpolates continuous feature vectors, whereas token-ID sequences are discrete and interpolated embeddings would not correspond to observed review texts.
