# Source-Grouped BERT Experiments

Status: **complete** (14/14 conditions complete).

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
| review_count | weighted_synthetic_reviews_2001 | 2001 | 0.3182 | 0.5321 | 0.3226 | 1h 45m |
| review_count | weighted_synthetic_reviews_2352 | 2352 | 0.3186 | 0.5320 | 0.3473 | 1h 36m |
| balancing | youtube_only_unbalanced | 0 | 0.3254 | 0.5327 | 0.3731 | 58m 37s |
| balancing | youtube_only_hybrid | 0 | 0.3551 | 0.5384 | 0.3285 | 19m 17s |
| balancing | youtube_plus_synthetic_784x3_unbalanced | 2352 | 0.3254 | 0.5185 | 0.3049 | 1h 48m |
| balancing | youtube_plus_synthetic_784x3_hybrid | 2352 | 0.3241 | 0.5330 | 0.2899 | 2h 13m |
| balancing | youtube_plus_synthetic_784x6_unbalanced | 4704 | 0.3254 | 0.5015 | 0.3179 | 2h 12m |
| balancing | youtube_plus_synthetic_784x6_hybrid | 4704 | 0.3228 | 0.5426 | 0.3249 | 2h 43m |

SMOTE: **not applicable**. Not applicable to raw-text BERT: SMOTE interpolates continuous feature vectors, whereas token-ID sequences are discrete and interpolated embeddings would not correspond to observed review texts.
