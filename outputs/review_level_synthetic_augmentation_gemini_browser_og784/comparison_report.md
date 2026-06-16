# Full Synthetic Review-Level Augmentation Comparison

- Model: Logistic Regression (`logistic_regression`)

## Metric Summary

| experiment | eval_rows | train_rows | macro_f1 | macro_roc_auc | macro_pr_auc | micro_f1 | hamming_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| og_only_reference | 784 | 784 | 0.3602 | 0.5630 | 0.3270 | 0.6471 | 0.2711 |
| synthetic_only_review_level | 2352 | 2352 | 0.6277 | 0.9424 | 0.6497 | 0.7807 | 0.1350 |
| og_plus_synthetic_all_rows_oof | 3136 | 3136 | 0.5849 | 0.9018 | 0.6294 | 0.7197 | 0.1853 |
| og_plus_synthetic_eval_og_rows | 784 | 3136 | 0.3744 | 0.5952 | 0.3312 | 0.6483 | 0.2542 |

## Label Distribution

| source | label | rows | positive_count | positive_rate |
| --- | --- | --- | --- | --- |
| og_784_perfume_level | joy | 784 | 622 | 0.7934 |
| og_784_perfume_level | sadness | 784 | 11 | 0.0140 |
| og_784_perfume_level | fear | 784 | 3 | 0.0038 |
| og_784_perfume_level | trust | 784 | 476 | 0.6071 |
| og_784_perfume_level | surprise | 784 | 44 | 0.0561 |
| og_784_perfume_level | anticipation | 784 | 366 | 0.4668 |
| og_784_perfume_level | disgust | 784 | 19 | 0.0242 |
| synthetic_v2_review_level | joy | 2352 | 312 | 0.1327 |
| synthetic_v2_review_level | sadness | 2352 | 159 | 0.0676 |
| synthetic_v2_review_level | fear | 2352 | 221 | 0.0940 |
| synthetic_v2_review_level | trust | 2352 | 834 | 0.3546 |
| synthetic_v2_review_level | surprise | 2352 | 774 | 0.3291 |
| synthetic_v2_review_level | anticipation | 2352 | 1834 | 0.7798 |
| synthetic_v2_review_level | disgust | 2352 | 23 | 0.0098 |
| og_plus_synthetic_rows | joy | 3136 | 934 | 0.2978 |
| og_plus_synthetic_rows | sadness | 3136 | 170 | 0.0542 |
| og_plus_synthetic_rows | fear | 3136 | 224 | 0.0714 |
| og_plus_synthetic_rows | trust | 3136 | 1310 | 0.4177 |
| og_plus_synthetic_rows | surprise | 3136 | 818 | 0.2608 |
| og_plus_synthetic_rows | anticipation | 3136 | 2200 | 0.7015 |
| og_plus_synthetic_rows | disgust | 3136 | 42 | 0.0134 |

## Interpretation Notes

- `og_only_reference` is the existing 784 perfume-level Logistic Regression result.
- `synthetic_only_review_level` is 2352 synthetic review rows, grouped by perfume.
- `og_plus_synthetic_all_rows_oof` evaluates the mixed row set, so synthetic rows dominate the metric 3:1.
- `og_plus_synthetic_eval_og_rows` is the thesis-relevant augmentation check: train with OG+synthetic rows, but evaluate only held-out OG perfume rows using the frozen fold assignment.
- This still mixes row granularities: OG rows are perfume-level, synthetic rows are review-level.
