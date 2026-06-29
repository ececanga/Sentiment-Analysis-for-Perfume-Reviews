# Synthetic-Only 784x6 Data-Level Resampling Ablation

This compares imbalanced-learn random oversampling, random undersampling, and SMOTE against unbalanced and model-weighted baselines.

- Rows: 4704
- Unique perfumes: 784
- Reviews per perfume: 6
- Review indices: 1, 2, 3, 4, 5, 6
- Threshold: 0.3
- imbalanced-learn version: 0.14.2
- Split: GroupKFold with 5 folds, grouped by perfume_id
- Resampling is done inside each training fold separately, for each binary label classifier.
- Test folds are never resampled.

## Metric Summary

| model_family | strategy | resampling_method | runs | macro_f1_mean | macro_f1_std | delta_macro_f1_mean_vs_unbalanced | macro_roc_auc_mean | macro_pr_auc_mean | predicted_label_cardinality_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | unbalanced | none | 5 | 0.5876 | 0.0000 | 0.0000 | 0.9006 | 0.5957 | 1.9526 |
| Logistic Regression | smote | smote | 5 | 0.5731 | 0.0007 | -0.0145 | 0.8978 | 0.5834 | 2.6187 |
| Logistic Regression | oversampling | oversampling | 5 | 0.5645 | 0.0008 | -0.0231 | 0.9009 | 0.5867 | 2.7282 |
| Logistic Regression | class_weight | none | 5 | 0.5640 | 0.0000 | -0.0236 | 0.9018 | 0.5884 | 2.7398 |
| Logistic Regression | undersampling | undersampling | 5 | 0.5507 | 0.0014 | -0.0369 | 0.8943 | 0.5676 | 2.8278 |
| XGBoost | scale_pos_weight | none | 5 | 0.5982 | 0.0021 | 0.0182 | 0.8994 | 0.5982 | 2.2153 |
| XGBoost | smote | smote | 5 | 0.5967 | 0.0019 | 0.0167 | 0.8984 | 0.5955 | 2.2612 |
| XGBoost | oversampling | oversampling | 5 | 0.5954 | 0.0026 | 0.0154 | 0.8994 | 0.5985 | 2.2068 |
| XGBoost | unbalanced | none | 5 | 0.5800 | 0.0018 | 0.0000 | 0.9015 | 0.5983 | 1.9026 |
| XGBoost | undersampling | undersampling | 5 | 0.5706 | 0.0016 | -0.0094 | 0.9001 | 0.5909 | 2.6633 |

## Per-Label Metrics

| model_family | strategy | label | positive_rate | f1_mean | delta_f1_mean_vs_unbalanced | pr_auc_mean | delta_pr_auc_mean_vs_unbalanced | predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | class_weight | anticipation | 0.6020 | 0.8185 | 0.0024 | 0.7776 | -0.0001 | 0.7963 |
| Logistic Regression | oversampling | anticipation | 0.6020 | 0.8188 | 0.0027 | 0.7778 | 0.0001 | 0.7970 |
| Logistic Regression | smote | anticipation | 0.6020 | 0.8186 | 0.0025 | 0.7784 | 0.0007 | 0.7968 |
| Logistic Regression | unbalanced | anticipation | 0.6020 | 0.8161 | 0.0000 | 0.7777 | 0.0000 | 0.8161 |
| Logistic Regression | undersampling | anticipation | 0.6020 | 0.8176 | 0.0015 | 0.7763 | -0.0014 | 0.7957 |
| Logistic Regression | class_weight | disgust | 0.0251 | 0.2220 | 0.1103 | 0.1516 | -0.0077 | 0.1818 |
| Logistic Regression | oversampling | disgust | 0.0251 | 0.2211 | 0.1093 | 0.1491 | -0.0102 | 0.1792 |
| Logistic Regression | smote | disgust | 0.0251 | 0.2263 | 0.1146 | 0.1467 | -0.0126 | 0.1609 |
| Logistic Regression | unbalanced | disgust | 0.0251 | 0.1117 | 0.0000 | 0.1593 | 0.0000 | 0.0130 |
| Logistic Regression | undersampling | disgust | 0.0251 | 0.1813 | 0.0696 | 0.1285 | -0.0309 | 0.2345 |
| Logistic Regression | class_weight | fear | 0.1135 | 0.6615 | -0.0771 | 0.7108 | -0.0009 | 0.2188 |
| Logistic Regression | oversampling | fear | 0.1135 | 0.6631 | -0.0756 | 0.7065 | -0.0053 | 0.2182 |
| Logistic Regression | smote | fear | 0.1135 | 0.6920 | -0.0467 | 0.7000 | -0.0118 | 0.2020 |
| Logistic Regression | unbalanced | fear | 0.1135 | 0.7387 | 0.0000 | 0.7117 | 0.0000 | 0.1541 |
| Logistic Regression | undersampling | fear | 0.1135 | 0.6524 | -0.0863 | 0.6772 | -0.0346 | 0.2231 |
| Logistic Regression | class_weight | joy | 0.0923 | 0.3608 | -0.2113 | 0.4944 | -0.0193 | 0.3744 |
| Logistic Regression | oversampling | joy | 0.0923 | 0.3626 | -0.2095 | 0.4930 | -0.0207 | 0.3702 |
| Logistic Regression | smote | joy | 0.0923 | 0.3819 | -0.1902 | 0.4662 | -0.0475 | 0.3281 |
| Logistic Regression | unbalanced | joy | 0.0923 | 0.5721 | 0.0000 | 0.5137 | 0.0000 | 0.0816 |
| Logistic Regression | undersampling | joy | 0.0923 | 0.3490 | -0.2231 | 0.4439 | -0.0699 | 0.3854 |
| Logistic Regression | class_weight | sadness | 0.0442 | 0.3487 | 0.0344 | 0.2632 | -0.0171 | 0.1996 |
| Logistic Regression | oversampling | sadness | 0.0442 | 0.3506 | 0.0363 | 0.2613 | -0.0190 | 0.1978 |
| Logistic Regression | smote | sadness | 0.0442 | 0.3544 | 0.0401 | 0.2746 | -0.0057 | 0.1835 |
| Logistic Regression | unbalanced | sadness | 0.0442 | 0.3144 | 0.0000 | 0.2802 | 0.0000 | 0.0491 |
| Logistic Regression | undersampling | sadness | 0.0442 | 0.3214 | 0.0070 | 0.2340 | -0.0462 | 0.2202 |
| Logistic Regression | class_weight | surprise | 0.3051 | 0.7883 | 0.0021 | 0.8393 | -0.0051 | 0.4592 |
| Logistic Regression | oversampling | surprise | 0.3051 | 0.7882 | 0.0020 | 0.8374 | -0.0069 | 0.4577 |
| Logistic Regression | smote | surprise | 0.3051 | 0.7873 | 0.0011 | 0.8358 | -0.0085 | 0.4529 |
| Logistic Regression | unbalanced | surprise | 0.3051 | 0.7862 | 0.0000 | 0.8443 | 0.0000 | 0.4169 |
| Logistic Regression | undersampling | surprise | 0.3051 | 0.7889 | 0.0027 | 0.8333 | -0.0110 | 0.4577 |
| Logistic Regression | class_weight | trust | 0.3506 | 0.7482 | -0.0258 | 0.8822 | -0.0010 | 0.5098 |
| Logistic Regression | oversampling | trust | 0.3506 | 0.7473 | -0.0268 | 0.8815 | -0.0017 | 0.5080 |
| Logistic Regression | smote | trust | 0.3506 | 0.7510 | -0.0230 | 0.8820 | -0.0012 | 0.4945 |
| Logistic Regression | unbalanced | trust | 0.3506 | 0.7740 | 0.0000 | 0.8832 | 0.0000 | 0.4218 |
| Logistic Regression | undersampling | trust | 0.3506 | 0.7444 | -0.0296 | 0.8799 | -0.0032 | 0.5113 |
| XGBoost | oversampling | anticipation | 0.6020 | 0.8207 | 0.0001 | 0.7693 | -0.0026 | 0.7978 |
| XGBoost | scale_pos_weight | anticipation | 0.6020 | 0.8210 | 0.0004 | 0.7713 | -0.0006 | 0.7960 |
| XGBoost | smote | anticipation | 0.6020 | 0.8193 | -0.0013 | 0.7657 | -0.0063 | 0.7972 |
| XGBoost | unbalanced | anticipation | 0.6020 | 0.8205 | 0.0000 | 0.7719 | 0.0000 | 0.8062 |
| XGBoost | undersampling | anticipation | 0.6020 | 0.8184 | -0.0022 | 0.7694 | -0.0026 | 0.7869 |
| XGBoost | oversampling | disgust | 0.0251 | 0.2367 | 0.1298 | 0.1759 | 0.0011 | 0.0489 |
| XGBoost | scale_pos_weight | disgust | 0.0251 | 0.2530 | 0.1461 | 0.1790 | 0.0043 | 0.0519 |
| XGBoost | smote | disgust | 0.0251 | 0.2655 | 0.1585 | 0.1760 | 0.0012 | 0.0716 |
| XGBoost | unbalanced | disgust | 0.0251 | 0.1069 | 0.0000 | 0.1748 | 0.0000 | 0.0123 |
| XGBoost | undersampling | disgust | 0.0251 | 0.2143 | 0.1073 | 0.1861 | 0.0113 | 0.1972 |
| XGBoost | oversampling | fear | 0.1135 | 0.7175 | -0.0094 | 0.6864 | 0.0052 | 0.1716 |
| XGBoost | scale_pos_weight | fear | 0.1135 | 0.7168 | -0.0101 | 0.6844 | 0.0033 | 0.1713 |
| XGBoost | smote | fear | 0.1135 | 0.7135 | -0.0134 | 0.6854 | 0.0042 | 0.1745 |
| XGBoost | unbalanced | fear | 0.1135 | 0.7269 | 0.0000 | 0.6811 | 0.0000 | 0.1454 |
| XGBoost | undersampling | fear | 0.1135 | 0.6783 | -0.0486 | 0.6626 | -0.0185 | 0.2082 |
| XGBoost | oversampling | joy | 0.0923 | 0.4649 | -0.0930 | 0.5348 | 0.0146 | 0.1900 |
| XGBoost | scale_pos_weight | joy | 0.0923 | 0.4640 | -0.0940 | 0.5327 | 0.0125 | 0.1869 |
| XGBoost | smote | joy | 0.0923 | 0.4446 | -0.1133 | 0.5242 | 0.0040 | 0.2096 |
| XGBoost | unbalanced | joy | 0.0923 | 0.5579 | 0.0000 | 0.5202 | 0.0000 | 0.0759 |
| XGBoost | undersampling | joy | 0.0923 | 0.3737 | -0.1842 | 0.5113 | -0.0089 | 0.3380 |
| XGBoost | oversampling | sadness | 0.0442 | 0.3619 | 0.0809 | 0.2715 | -0.0159 | 0.1085 |
| XGBoost | scale_pos_weight | sadness | 0.0442 | 0.3670 | 0.0861 | 0.2681 | -0.0193 | 0.1077 |
| XGBoost | smote | sadness | 0.0442 | 0.3700 | 0.0890 | 0.2708 | -0.0166 | 0.1164 |
| XGBoost | unbalanced | sadness | 0.0442 | 0.2810 | 0.0000 | 0.2874 | 0.0000 | 0.0362 |
| XGBoost | undersampling | sadness | 0.0442 | 0.3553 | 0.0743 | 0.2606 | -0.0268 | 0.1956 |
| XGBoost | oversampling | surprise | 0.3051 | 0.7890 | 0.0061 | 0.8573 | -0.0012 | 0.4417 |
| XGBoost | scale_pos_weight | surprise | 0.3051 | 0.7902 | 0.0073 | 0.8571 | -0.0013 | 0.4464 |
| XGBoost | smote | surprise | 0.3051 | 0.7880 | 0.0051 | 0.8543 | -0.0041 | 0.4427 |
| XGBoost | unbalanced | surprise | 0.3051 | 0.7829 | 0.0000 | 0.8584 | 0.0000 | 0.4211 |
| XGBoost | undersampling | surprise | 0.3051 | 0.7919 | 0.0090 | 0.8555 | -0.0029 | 0.4541 |
| XGBoost | oversampling | trust | 0.3506 | 0.7770 | -0.0067 | 0.8943 | 0.0003 | 0.4483 |
| XGBoost | scale_pos_weight | trust | 0.3506 | 0.7757 | -0.0080 | 0.8945 | 0.0006 | 0.4551 |
| XGBoost | smote | trust | 0.3506 | 0.7758 | -0.0080 | 0.8921 | -0.0019 | 0.4492 |
| XGBoost | unbalanced | trust | 0.3506 | 0.7837 | 0.0000 | 0.8940 | 0.0000 | 0.4056 |
| XGBoost | undersampling | trust | 0.3506 | 0.7623 | -0.0214 | 0.8909 | -0.0031 | 0.4833 |

## Label Distribution

| label | positive_count | positive_rate | neg_to_pos_ratio |
| --- | --- | --- | --- |
| joy | 434 | 0.0923 | 9.8387 |
| sadness | 208 | 0.0442 | 21.6154 |
| fear | 534 | 0.1135 | 7.8090 |
| trust | 1649 | 0.3506 | 1.8526 |
| surprise | 1435 | 0.3051 | 2.2780 |
| anticipation | 2832 | 0.6020 | 0.6610 |
| disgust | 118 | 0.0251 | 38.8644 |
