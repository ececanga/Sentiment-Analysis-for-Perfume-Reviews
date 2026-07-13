# OG YouTube + Synthetic Balancing Ablation

Main metric: held-out OG YouTube rows using the frozen OG fold assignments.

- OG rows: 784
- Synthetic 784x3 rows: 2352
- Synthetic 784x6 rows: 4704
- imbalanced-learn version: 0.14.2

## Balanced-Technique Winners On Held-Out OG Rows

| threshold | data_label | model_family | strategy | macro_f1 | macro_roc_auc | macro_pr_auc | delta_macro_f1_vs_unbalanced | predicted_label_cardinality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3000 | OG YouTube only | XGBoost | smote | 0.3404 | 0.5668 | 0.3150 | 0.0162 | 2.5931 |
| 0.3000 | OG YouTube + synthetic 784x3 | XGBoost | smote | 0.3258 | 0.6013 | 0.3059 | -0.0045 | 2.5497 |
| 0.3000 | OG YouTube + synthetic 784x6 | XGBoost | hybrid | 0.3794 | 0.5900 | 0.3531 | 0.0497 | 2.6033 |

## Held-Out OG Row Metrics

`unbalanced` is included only as a baseline reference.

| threshold | data_label | model_family | strategy | macro_f1 | macro_roc_auc | macro_pr_auc | delta_macro_f1_vs_unbalanced | predicted_label_cardinality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3000 | OG YouTube only | XGBoost | smote | 0.3404 | 0.5668 | 0.3150 | 0.0162 | 2.5931 |
| 0.3000 | OG YouTube only | XGBoost | unbalanced | 0.3242 | 0.5351 | 0.2979 | 0.0000 | 2.5855 |
| 0.3000 | OG YouTube only | XGBoost | hybrid | 0.2230 | 0.5328 | 0.3049 | -0.1012 | 1.5459 |
| 0.3000 | OG YouTube + synthetic 784x3 | XGBoost | unbalanced | 0.3303 | 0.5770 | 0.3086 | 0.0000 | 2.7028 |
| 0.3000 | OG YouTube + synthetic 784x3 | XGBoost | smote | 0.3258 | 0.6013 | 0.3059 | -0.0045 | 2.5497 |
| 0.3000 | OG YouTube + synthetic 784x3 | XGBoost | hybrid | 0.3140 | 0.5669 | 0.3131 | -0.0164 | 2.2870 |
| 0.3000 | OG YouTube + synthetic 784x6 | XGBoost | hybrid | 0.3794 | 0.5900 | 0.3531 | 0.0497 | 2.6033 |
| 0.3000 | OG YouTube + synthetic 784x6 | XGBoost | smote | 0.3517 | 0.5819 | 0.3074 | 0.0219 | 2.8074 |
| 0.3000 | OG YouTube + synthetic 784x6 | XGBoost | unbalanced | 0.3297 | 0.6051 | 0.3651 | 0.0000 | 2.8099 |

## Label Distribution

| source | label | rows | positive_count | positive_rate |
| --- | --- | --- | --- | --- |
| og_youtube_rows | joy | 784 | 622 | 0.7934 |
| og_youtube_rows | sadness | 784 | 11 | 0.0140 |
| og_youtube_rows | fear | 784 | 3 | 0.0038 |
| og_youtube_rows | trust | 784 | 476 | 0.6071 |
| og_youtube_rows | surprise | 784 | 44 | 0.0561 |
| og_youtube_rows | anticipation | 784 | 366 | 0.4668 |
| og_youtube_rows | disgust | 784 | 19 | 0.0242 |
| synthetic_784x3_rows | joy | 2352 | 312 | 0.1327 |
| synthetic_784x3_rows | sadness | 2352 | 159 | 0.0676 |
| synthetic_784x3_rows | fear | 2352 | 221 | 0.0940 |
| synthetic_784x3_rows | trust | 2352 | 834 | 0.3546 |
| synthetic_784x3_rows | surprise | 2352 | 774 | 0.3291 |
| synthetic_784x3_rows | anticipation | 2352 | 1834 | 0.7798 |
| synthetic_784x3_rows | disgust | 2352 | 23 | 0.0098 |
| synthetic_784x6_rows | joy | 4704 | 434 | 0.0923 |
| synthetic_784x6_rows | sadness | 4704 | 208 | 0.0442 |
| synthetic_784x6_rows | fear | 4704 | 534 | 0.1135 |
| synthetic_784x6_rows | trust | 4704 | 1649 | 0.3506 |
| synthetic_784x6_rows | surprise | 4704 | 1435 | 0.3051 |
| synthetic_784x6_rows | anticipation | 4704 | 2832 | 0.6020 |
| synthetic_784x6_rows | disgust | 4704 | 118 | 0.0251 |
| og_plus_synthetic_3reviews | joy | 3136 | 934 | 0.2978 |
| og_plus_synthetic_3reviews | sadness | 3136 | 170 | 0.0542 |
| og_plus_synthetic_3reviews | fear | 3136 | 224 | 0.0714 |
| og_plus_synthetic_3reviews | trust | 3136 | 1310 | 0.4177 |
| og_plus_synthetic_3reviews | surprise | 3136 | 818 | 0.2608 |
| og_plus_synthetic_3reviews | anticipation | 3136 | 2200 | 0.7015 |
| og_plus_synthetic_3reviews | disgust | 3136 | 42 | 0.0134 |
| og_plus_synthetic_6reviews | joy | 5488 | 1056 | 0.1924 |
| og_plus_synthetic_6reviews | sadness | 5488 | 219 | 0.0399 |
| og_plus_synthetic_6reviews | fear | 5488 | 537 | 0.0978 |
| og_plus_synthetic_6reviews | trust | 5488 | 2125 | 0.3872 |
| og_plus_synthetic_6reviews | surprise | 5488 | 1479 | 0.2695 |
| og_plus_synthetic_6reviews | anticipation | 5488 | 3198 | 0.5827 |
| og_plus_synthetic_6reviews | disgust | 5488 | 137 | 0.0250 |

## Notes

- Secondary all-row metrics are saved separately because synthetic rows dominate mixed row counts.
- Resampling is applied only inside each training fold and separately per binary emotion label.
- Evaluation folds exclude every row from the held-out perfume IDs, including synthetic rows for those perfumes.
