# Synthetic-Only 784x3 Data-Level Resampling Ablation

This compares imbalanced-learn random oversampling, random undersampling, and SMOTE against unbalanced and model-weighted baselines.

- Rows: 2352
- Unique perfumes: 784
- Reviews per perfume: 3
- Review indices: 1, 2, 3
- Threshold: 0.5
- imbalanced-learn version: 0.14.2
- Split: GroupKFold with 5 folds, grouped by perfume_id
- Resampling is done inside each training fold separately, for each binary label classifier.
- Test folds are never resampled.

## Metric Summary

| model_family | strategy | resampling_method | runs | macro_f1_mean | macro_f1_std | delta_macro_f1_mean_vs_unbalanced | macro_roc_auc_mean | macro_pr_auc_mean | predicted_label_cardinality_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | class_weight | none | 5 | 0.6487 | 0.0000 | 0.0572 | 0.9424 | 0.6497 | 2.1998 |
| Logistic Regression | oversampling | oversampling | 5 | 0.6470 | 0.0023 | 0.0555 | 0.9419 | 0.6459 | 2.1728 |
| Logistic Regression | smote | smote | 5 | 0.6390 | 0.0020 | 0.0475 | 0.9331 | 0.6443 | 2.1233 |
| Logistic Regression | undersampling | undersampling | 5 | 0.6117 | 0.0057 | 0.0201 | 0.9312 | 0.6073 | 2.3455 |
| Logistic Regression | unbalanced | none | 5 | 0.5915 | 0.0000 | 0.0000 | 0.9428 | 0.6598 | 1.7156 |
| XGBoost | scale_pos_weight | none | 5 | 0.6585 | 0.0032 | 0.0450 | 0.9483 | 0.6745 | 1.8901 |
| XGBoost | smote | smote | 5 | 0.6544 | 0.0034 | 0.0409 | 0.9470 | 0.6766 | 1.9438 |
| XGBoost | oversampling | oversampling | 5 | 0.6485 | 0.0046 | 0.0350 | 0.9482 | 0.6719 | 1.8827 |
| XGBoost | undersampling | undersampling | 5 | 0.6290 | 0.0050 | 0.0155 | 0.9449 | 0.6649 | 2.3022 |
| XGBoost | unbalanced | none | 5 | 0.6135 | 0.0042 | 0.0000 | 0.9498 | 0.6822 | 1.7224 |

## Per-Label Metrics

| model_family | strategy | label | positive_rate | f1_mean | delta_f1_mean_vs_unbalanced | pr_auc_mean | delta_pr_auc_mean_vs_unbalanced | predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | class_weight | anticipation | 0.7798 | 0.9291 | -0.0263 | 0.9876 | -0.0000 | 0.7258 |
| Logistic Regression | oversampling | anticipation | 0.7798 | 0.9292 | -0.0262 | 0.9875 | -0.0001 | 0.7240 |
| Logistic Regression | smote | anticipation | 0.7798 | 0.9278 | -0.0276 | 0.9869 | -0.0007 | 0.7253 |
| Logistic Regression | unbalanced | anticipation | 0.7798 | 0.9554 | 0.0000 | 0.9876 | 0.0000 | 0.8036 |
| Logistic Regression | undersampling | anticipation | 0.7798 | 0.9203 | -0.0351 | 0.9863 | -0.0013 | 0.7139 |
| Logistic Regression | class_weight | disgust | 0.0098 | 0.2517 | 0.2517 | 0.1314 | -0.0305 | 0.0510 |
| Logistic Regression | oversampling | disgust | 0.0098 | 0.2426 | 0.2426 | 0.1319 | -0.0299 | 0.0421 |
| Logistic Regression | smote | disgust | 0.0098 | 0.1812 | 0.1812 | 0.1123 | -0.0496 | 0.0418 |
| Logistic Regression | unbalanced | disgust | 0.0098 | 0.0000 | 0.0000 | 0.1619 | 0.0000 | 0.0013 |
| Logistic Regression | undersampling | disgust | 0.0098 | 0.1170 | 0.1170 | 0.0665 | -0.0954 | 0.1492 |
| Logistic Regression | class_weight | fear | 0.0940 | 0.6945 | 0.0380 | 0.6367 | 0.0077 | 0.1607 |
| Logistic Regression | oversampling | fear | 0.0940 | 0.6977 | 0.0412 | 0.6260 | -0.0030 | 0.1583 |
| Logistic Regression | smote | fear | 0.0940 | 0.7107 | 0.0543 | 0.6276 | -0.0014 | 0.1518 |
| Logistic Regression | unbalanced | fear | 0.0940 | 0.6565 | 0.0000 | 0.6290 | 0.0000 | 0.1003 |
| Logistic Regression | undersampling | fear | 0.0940 | 0.6472 | -0.0093 | 0.5522 | -0.0768 | 0.1791 |
| Logistic Regression | class_weight | joy | 0.1327 | 0.5225 | -0.0791 | 0.6086 | -0.0240 | 0.2645 |
| Logistic Regression | oversampling | joy | 0.1327 | 0.5208 | -0.0808 | 0.6031 | -0.0295 | 0.2582 |
| Logistic Regression | smote | joy | 0.1327 | 0.5288 | -0.0727 | 0.5888 | -0.0437 | 0.2449 |
| Logistic Regression | unbalanced | joy | 0.1327 | 0.6016 | 0.0000 | 0.6326 | 0.0000 | 0.0850 |
| Logistic Regression | undersampling | joy | 0.1327 | 0.5024 | -0.0992 | 0.5388 | -0.0938 | 0.2871 |
| Logistic Regression | class_weight | sadness | 0.0676 | 0.5000 | 0.1602 | 0.3996 | -0.0203 | 0.1790 |
| Logistic Regression | oversampling | sadness | 0.0676 | 0.5004 | 0.1606 | 0.3912 | -0.0287 | 0.1751 |
| Logistic Regression | smote | sadness | 0.0676 | 0.4926 | 0.1528 | 0.4126 | -0.0073 | 0.1627 |
| Logistic Regression | unbalanced | sadness | 0.0676 | 0.3398 | 0.0000 | 0.4199 | 0.0000 | 0.0425 |
| Logistic Regression | undersampling | sadness | 0.0676 | 0.4597 | 0.1199 | 0.3317 | -0.0882 | 0.1981 |
| Logistic Regression | class_weight | surprise | 0.3291 | 0.8127 | 0.0348 | 0.8791 | -0.0024 | 0.4179 |
| Logistic Regression | oversampling | surprise | 0.3291 | 0.8126 | 0.0346 | 0.8783 | -0.0031 | 0.4164 |
| Logistic Regression | smote | surprise | 0.3291 | 0.8084 | 0.0304 | 0.8772 | -0.0043 | 0.4049 |
| Logistic Regression | unbalanced | surprise | 0.3291 | 0.7780 | 0.0000 | 0.8815 | 0.0000 | 0.3431 |
| Logistic Regression | undersampling | surprise | 0.3291 | 0.8085 | 0.0305 | 0.8725 | -0.0090 | 0.4187 |
| Logistic Regression | class_weight | trust | 0.3546 | 0.8306 | 0.0211 | 0.9049 | -0.0015 | 0.4009 |
| Logistic Regression | oversampling | trust | 0.3546 | 0.8259 | 0.0164 | 0.9031 | -0.0033 | 0.3986 |
| Logistic Regression | smote | trust | 0.3546 | 0.8236 | 0.0141 | 0.9051 | -0.0013 | 0.3918 |
| Logistic Regression | unbalanced | trust | 0.3546 | 0.8096 | 0.0000 | 0.9064 | 0.0000 | 0.3397 |
| Logistic Regression | undersampling | trust | 0.3546 | 0.8265 | 0.0170 | 0.9029 | -0.0036 | 0.3993 |
| XGBoost | oversampling | anticipation | 0.7798 | 0.9576 | -0.0027 | 0.9913 | -0.0003 | 0.7832 |
| XGBoost | scale_pos_weight | anticipation | 0.7798 | 0.9571 | -0.0031 | 0.9915 | -0.0000 | 0.7798 |
| XGBoost | smote | anticipation | 0.7798 | 0.9542 | -0.0061 | 0.9915 | -0.0001 | 0.7737 |
| XGBoost | unbalanced | anticipation | 0.7798 | 0.9603 | 0.0000 | 0.9916 | 0.0000 | 0.7982 |
| XGBoost | undersampling | anticipation | 0.7798 | 0.9324 | -0.0279 | 0.9909 | -0.0007 | 0.7243 |
| XGBoost | oversampling | disgust | 0.0098 | 0.1386 | 0.0633 | 0.1428 | -0.0612 | 0.0099 |
| XGBoost | scale_pos_weight | disgust | 0.0098 | 0.1862 | 0.1110 | 0.1756 | -0.0283 | 0.0093 |
| XGBoost | smote | disgust | 0.0098 | 0.2175 | 0.1423 | 0.1993 | -0.0047 | 0.0191 |
| XGBoost | unbalanced | disgust | 0.0098 | 0.0753 | 0.0000 | 0.2040 | 0.0000 | 0.0015 |
| XGBoost | undersampling | disgust | 0.0098 | 0.1461 | 0.0708 | 0.1946 | -0.0094 | 0.1163 |
| XGBoost | oversampling | fear | 0.0940 | 0.7022 | 0.0291 | 0.6417 | -0.0076 | 0.1213 |
| XGBoost | scale_pos_weight | fear | 0.0940 | 0.7132 | 0.0401 | 0.6361 | -0.0132 | 0.1216 |
| XGBoost | smote | fear | 0.0940 | 0.7062 | 0.0331 | 0.6558 | 0.0065 | 0.1286 |
| XGBoost | unbalanced | fear | 0.0940 | 0.6731 | 0.0000 | 0.6493 | 0.0000 | 0.0988 |
| XGBoost | undersampling | fear | 0.0940 | 0.6634 | -0.0096 | 0.6072 | -0.0421 | 0.1732 |
| XGBoost | oversampling | joy | 0.1327 | 0.6124 | 0.0082 | 0.6777 | -0.0015 | 0.1203 |
| XGBoost | scale_pos_weight | joy | 0.1327 | 0.6133 | 0.0091 | 0.6787 | -0.0004 | 0.1189 |
| XGBoost | smote | joy | 0.1327 | 0.5908 | -0.0134 | 0.6651 | -0.0140 | 0.1512 |
| XGBoost | unbalanced | joy | 0.1327 | 0.6041 | 0.0000 | 0.6791 | 0.0000 | 0.0804 |
| XGBoost | undersampling | joy | 0.1327 | 0.5293 | -0.0749 | 0.6441 | -0.0350 | 0.2700 |
| XGBoost | oversampling | sadness | 0.0676 | 0.4977 | 0.1299 | 0.4409 | -0.0029 | 0.0934 |
| XGBoost | scale_pos_weight | sadness | 0.0676 | 0.4978 | 0.1300 | 0.4319 | -0.0119 | 0.0940 |
| XGBoost | smote | sadness | 0.0676 | 0.4781 | 0.1103 | 0.4177 | -0.0260 | 0.1053 |
| XGBoost | unbalanced | sadness | 0.0676 | 0.3678 | 0.0000 | 0.4438 | 0.0000 | 0.0419 |
| XGBoost | undersampling | sadness | 0.0676 | 0.4905 | 0.1227 | 0.4145 | -0.0293 | 0.1889 |
| XGBoost | oversampling | surprise | 0.3291 | 0.8081 | 0.0134 | 0.8841 | 0.0011 | 0.3894 |
| XGBoost | scale_pos_weight | surprise | 0.3291 | 0.8122 | 0.0175 | 0.8819 | -0.0011 | 0.3960 |
| XGBoost | smote | surprise | 0.3291 | 0.8112 | 0.0164 | 0.8846 | 0.0016 | 0.3946 |
| XGBoost | unbalanced | surprise | 0.3291 | 0.7948 | 0.0000 | 0.8830 | 0.0000 | 0.3571 |
| XGBoost | undersampling | surprise | 0.3291 | 0.8166 | 0.0218 | 0.8806 | -0.0024 | 0.4284 |
| XGBoost | oversampling | trust | 0.3546 | 0.8232 | 0.0038 | 0.9246 | -0.0003 | 0.3652 |
| XGBoost | scale_pos_weight | trust | 0.3546 | 0.8295 | 0.0102 | 0.9258 | 0.0009 | 0.3706 |
| XGBoost | smote | trust | 0.3546 | 0.8229 | 0.0035 | 0.9220 | -0.0028 | 0.3713 |
| XGBoost | unbalanced | trust | 0.3546 | 0.8193 | 0.0000 | 0.9249 | 0.0000 | 0.3443 |
| XGBoost | undersampling | trust | 0.3546 | 0.8249 | 0.0056 | 0.9225 | -0.0023 | 0.4010 |

## Label Distribution

| label | positive_count | positive_rate | neg_to_pos_ratio |
| --- | --- | --- | --- |
| joy | 312 | 0.1327 | 6.5385 |
| sadness | 159 | 0.0676 | 13.7925 |
| fear | 221 | 0.0940 | 9.6425 |
| trust | 834 | 0.3546 | 1.8201 |
| surprise | 774 | 0.3291 | 2.0388 |
| anticipation | 1834 | 0.7798 | 0.2824 |
| disgust | 23 | 0.0098 | 101.2609 |
