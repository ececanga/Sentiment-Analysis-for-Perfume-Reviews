# OG YouTube + Synthetic Balancing Ablation

Main metric: held-out OG YouTube rows using the frozen OG fold assignments.

- OG rows: 784
- Synthetic 784x3 rows: 2352
- Synthetic 784x6 rows: 4704
- imbalanced-learn version: 0.14.2

## Balanced-Technique Winners On Held-Out OG Rows

| threshold | data_label | model_family | strategy | macro_f1 | macro_roc_auc | macro_pr_auc | delta_macro_f1_vs_unbalanced | predicted_label_cardinality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3000 | OG YouTube only | Logistic Regression | smote | 0.3812 | 0.6066 | 0.3295 | -0.0152 | 3.2194 |
| 0.3000 | OG YouTube only | XGBoost | scale_pos_weight | 0.3793 | 0.6224 | 0.3470 | 0.0192 | 2.5663 |
| 0.3000 | OG YouTube + synthetic 784x3 | Logistic Regression | smote | 0.3787 | 0.5913 | 0.3265 | -0.0344 | 2.9107 |
| 0.3000 | OG YouTube + synthetic 784x3 | XGBoost | undersampling | 0.4009 | 0.6338 | 0.3515 | 0.0399 | 3.1594 |
| 0.3000 | OG YouTube + synthetic 784x6 | Logistic Regression | smote | 0.4051 | 0.5866 | 0.3191 | 0.0524 | 3.3355 |
| 0.3000 | OG YouTube + synthetic 784x6 | XGBoost | undersampling | 0.3892 | 0.6358 | 0.3498 | -0.0278 | 3.3967 |
| 0.5000 | OG YouTube only | Logistic Regression | smote | 0.3392 | 0.6066 | 0.3295 | -0.0175 | 2.1518 |
| 0.5000 | OG YouTube only | XGBoost | smote | 0.3447 | 0.6378 | 0.3484 | 0.0431 | 1.9554 |
| 0.5000 | OG YouTube + synthetic 784x3 | Logistic Regression | smote | 0.3426 | 0.5913 | 0.3265 | 0.0387 | 2.1837 |
| 0.5000 | OG YouTube + synthetic 784x3 | XGBoost | oversampling | 0.3411 | 0.5938 | 0.3512 | 0.0382 | 2.0230 |
| 0.5000 | OG YouTube + synthetic 784x6 | Logistic Regression | oversampling | 0.3694 | 0.5851 | 0.3279 | 0.0696 | 2.3673 |
| 0.5000 | OG YouTube + synthetic 784x6 | XGBoost | undersampling | 0.3353 | 0.6358 | 0.3498 | 0.0450 | 2.5434 |

## Held-Out OG Row Metrics

`unbalanced` is included only as a baseline reference.

| threshold | data_label | model_family | strategy | macro_f1 | macro_roc_auc | macro_pr_auc | delta_macro_f1_vs_unbalanced | predicted_label_cardinality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3000 | OG YouTube only | Logistic Regression | unbalanced | 0.3964 | 0.5711 | 0.3387 | 0.0000 | 2.8048 |
| 0.3000 | OG YouTube only | Logistic Regression | smote | 0.3812 | 0.6066 | 0.3295 | -0.0152 | 3.2194 |
| 0.3000 | OG YouTube only | Logistic Regression | oversampling | 0.3754 | 0.5948 | 0.3252 | -0.0210 | 3.3265 |
| 0.3000 | OG YouTube only | Logistic Regression | class_weight | 0.3602 | 0.5630 | 0.3270 | -0.0362 | 3.4120 |
| 0.3000 | OG YouTube only | Logistic Regression | undersampling | 0.3410 | 0.5091 | 0.3100 | -0.0554 | 4.6288 |
| 0.3000 | OG YouTube only | XGBoost | scale_pos_weight | 0.3793 | 0.6224 | 0.3470 | 0.0192 | 2.5663 |
| 0.3000 | OG YouTube only | XGBoost | smote | 0.3695 | 0.6378 | 0.3484 | 0.0094 | 2.5982 |
| 0.3000 | OG YouTube only | XGBoost | oversampling | 0.3617 | 0.6313 | 0.3414 | 0.0016 | 2.5829 |
| 0.3000 | OG YouTube only | XGBoost | unbalanced | 0.3601 | 0.6015 | 0.3396 | 0.0000 | 2.5893 |
| 0.3000 | OG YouTube only | XGBoost | undersampling | 0.3271 | 0.5856 | 0.3104 | -0.0331 | 5.2309 |
| 0.3000 | OG YouTube + synthetic 784x3 | Logistic Regression | unbalanced | 0.4131 | 0.6297 | 0.3377 | 0.0000 | 2.7474 |
| 0.3000 | OG YouTube + synthetic 784x3 | Logistic Regression | smote | 0.3787 | 0.5913 | 0.3265 | -0.0344 | 2.9107 |
| 0.3000 | OG YouTube + synthetic 784x3 | Logistic Regression | oversampling | 0.3783 | 0.5982 | 0.3268 | -0.0349 | 2.9643 |
| 0.3000 | OG YouTube + synthetic 784x3 | Logistic Regression | class_weight | 0.3744 | 0.5952 | 0.3312 | -0.0388 | 3.0944 |
| 0.3000 | OG YouTube + synthetic 784x3 | Logistic Regression | undersampling | 0.3630 | 0.6301 | 0.3299 | -0.0502 | 3.5855 |
| 0.3000 | OG YouTube + synthetic 784x3 | XGBoost | undersampling | 0.4009 | 0.6338 | 0.3515 | 0.0399 | 3.1594 |
| 0.3000 | OG YouTube + synthetic 784x3 | XGBoost | scale_pos_weight | 0.3725 | 0.6381 | 0.3580 | 0.0115 | 2.5548 |
| 0.3000 | OG YouTube + synthetic 784x3 | XGBoost | oversampling | 0.3694 | 0.5938 | 0.3512 | 0.0085 | 2.5689 |
| 0.3000 | OG YouTube + synthetic 784x3 | XGBoost | unbalanced | 0.3609 | 0.6292 | 0.3589 | 0.0000 | 2.7219 |
| 0.3000 | OG YouTube + synthetic 784x3 | XGBoost | smote | 0.3445 | 0.6273 | 0.3680 | -0.0164 | 2.5918 |
| 0.3000 | OG YouTube + synthetic 784x6 | Logistic Regression | smote | 0.4051 | 0.5866 | 0.3191 | 0.0524 | 3.3355 |
| 0.3000 | OG YouTube + synthetic 784x6 | Logistic Regression | oversampling | 0.3963 | 0.5851 | 0.3279 | 0.0437 | 3.4605 |
| 0.3000 | OG YouTube + synthetic 784x6 | Logistic Regression | class_weight | 0.3845 | 0.5777 | 0.3133 | 0.0318 | 3.5064 |
| 0.3000 | OG YouTube + synthetic 784x6 | Logistic Regression | undersampling | 0.3697 | 0.5540 | 0.3062 | 0.0171 | 3.8495 |
| 0.3000 | OG YouTube + synthetic 784x6 | Logistic Regression | unbalanced | 0.3526 | 0.5905 | 0.3244 | 0.0000 | 2.8138 |
| 0.3000 | OG YouTube + synthetic 784x6 | XGBoost | unbalanced | 0.4171 | 0.6426 | 0.4136 | 0.0000 | 2.8316 |
| 0.3000 | OG YouTube + synthetic 784x6 | XGBoost | undersampling | 0.3892 | 0.6358 | 0.3498 | -0.0278 | 3.3967 |
| 0.3000 | OG YouTube + synthetic 784x6 | XGBoost | oversampling | 0.3670 | 0.6309 | 0.3686 | -0.0501 | 2.8036 |
| 0.3000 | OG YouTube + synthetic 784x6 | XGBoost | smote | 0.3662 | 0.6182 | 0.3403 | -0.0508 | 2.8559 |
| 0.3000 | OG YouTube + synthetic 784x6 | XGBoost | scale_pos_weight | 0.3642 | 0.6352 | 0.3633 | -0.0529 | 2.7781 |
| 0.5000 | OG YouTube only | Logistic Regression | unbalanced | 0.3567 | 0.5711 | 0.3387 | 0.0000 | 2.1773 |
| 0.5000 | OG YouTube only | Logistic Regression | smote | 0.3392 | 0.6066 | 0.3295 | -0.0175 | 2.1518 |
| 0.5000 | OG YouTube only | Logistic Regression | oversampling | 0.3336 | 0.5948 | 0.3252 | -0.0231 | 2.1735 |
| 0.5000 | OG YouTube only | Logistic Regression | class_weight | 0.3186 | 0.5630 | 0.3270 | -0.0381 | 2.2028 |
| 0.5000 | OG YouTube only | Logistic Regression | undersampling | 0.2872 | 0.5091 | 0.3100 | -0.0695 | 3.2602 |
| 0.5000 | OG YouTube only | XGBoost | smote | 0.3447 | 0.6378 | 0.3484 | 0.0431 | 1.9554 |
| 0.5000 | OG YouTube only | XGBoost | oversampling | 0.3372 | 0.6313 | 0.3414 | 0.0356 | 2.0179 |
| 0.5000 | OG YouTube only | XGBoost | scale_pos_weight | 0.3267 | 0.6224 | 0.3470 | 0.0251 | 2.0217 |
| 0.5000 | OG YouTube only | XGBoost | unbalanced | 0.3016 | 0.6015 | 0.3396 | 0.0000 | 2.1097 |
| 0.5000 | OG YouTube only | XGBoost | undersampling | 0.2941 | 0.5856 | 0.3104 | -0.0076 | 3.3278 |
| 0.5000 | OG YouTube + synthetic 784x3 | Logistic Regression | smote | 0.3426 | 0.5913 | 0.3265 | 0.0387 | 2.1837 |
| 0.5000 | OG YouTube + synthetic 784x3 | Logistic Regression | oversampling | 0.3376 | 0.5982 | 0.3268 | 0.0338 | 2.2105 |
| 0.5000 | OG YouTube + synthetic 784x3 | Logistic Regression | class_weight | 0.3286 | 0.5952 | 0.3312 | 0.0248 | 2.2296 |
| 0.5000 | OG YouTube + synthetic 784x3 | Logistic Regression | undersampling | 0.3165 | 0.6301 | 0.3299 | 0.0126 | 2.7500 |
| 0.5000 | OG YouTube + synthetic 784x3 | Logistic Regression | unbalanced | 0.3039 | 0.6297 | 0.3377 | 0.0000 | 2.0140 |
| 0.5000 | OG YouTube + synthetic 784x3 | XGBoost | oversampling | 0.3411 | 0.5938 | 0.3512 | 0.0382 | 2.0230 |
| 0.5000 | OG YouTube + synthetic 784x3 | XGBoost | undersampling | 0.3334 | 0.6338 | 0.3515 | 0.0305 | 2.5115 |
| 0.5000 | OG YouTube + synthetic 784x3 | XGBoost | smote | 0.3253 | 0.6273 | 0.3680 | 0.0224 | 2.0587 |
| 0.5000 | OG YouTube + synthetic 784x3 | XGBoost | scale_pos_weight | 0.3247 | 0.6381 | 0.3580 | 0.0218 | 2.0153 |
| 0.5000 | OG YouTube + synthetic 784x3 | XGBoost | unbalanced | 0.3029 | 0.6292 | 0.3589 | 0.0000 | 2.1327 |
| 0.5000 | OG YouTube + synthetic 784x6 | Logistic Regression | oversampling | 0.3694 | 0.5851 | 0.3279 | 0.0696 | 2.3673 |
| 0.5000 | OG YouTube + synthetic 784x6 | Logistic Regression | class_weight | 0.3453 | 0.5777 | 0.3133 | 0.0455 | 2.3776 |
| 0.5000 | OG YouTube + synthetic 784x6 | Logistic Regression | undersampling | 0.3289 | 0.5540 | 0.3062 | 0.0291 | 2.7181 |
| 0.5000 | OG YouTube + synthetic 784x6 | Logistic Regression | smote | 0.3193 | 0.5866 | 0.3191 | 0.0195 | 2.2959 |
| 0.5000 | OG YouTube + synthetic 784x6 | Logistic Regression | unbalanced | 0.2998 | 0.5905 | 0.3244 | 0.0000 | 1.9643 |
| 0.5000 | OG YouTube + synthetic 784x6 | XGBoost | undersampling | 0.3353 | 0.6358 | 0.3498 | 0.0450 | 2.5434 |
| 0.5000 | OG YouTube + synthetic 784x6 | XGBoost | oversampling | 0.3278 | 0.6309 | 0.3686 | 0.0376 | 2.0969 |
| 0.5000 | OG YouTube + synthetic 784x6 | XGBoost | scale_pos_weight | 0.3122 | 0.6352 | 0.3633 | 0.0219 | 2.1161 |
| 0.5000 | OG YouTube + synthetic 784x6 | XGBoost | smote | 0.3106 | 0.6182 | 0.3403 | 0.0204 | 2.1161 |
| 0.5000 | OG YouTube + synthetic 784x6 | XGBoost | unbalanced | 0.2902 | 0.6426 | 0.4136 | 0.0000 | 2.0804 |

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
