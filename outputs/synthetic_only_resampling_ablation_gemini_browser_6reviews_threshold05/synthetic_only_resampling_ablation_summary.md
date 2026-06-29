# Synthetic-Only 784x6 Data-Level Resampling Ablation

This compares imbalanced-learn random oversampling, random undersampling, and SMOTE against unbalanced and model-weighted baselines.

- Rows: 4704
- Unique perfumes: 784
- Review indices: 1, 2, 3, 4, 5, 6
- Threshold: 0.5
- imbalanced-learn version: 0.14.2
- Split: GroupKFold with 5 folds, grouped by perfume_id
- Resampling is done inside each training fold separately, for each binary label classifier.
- Test folds are never resampled.

## Metric Summary

| model_family | strategy | resampling_method | runs | macro_f1_mean | macro_f1_std | delta_macro_f1_mean_vs_unbalanced | macro_roc_auc_mean | macro_pr_auc_mean | predicted_label_cardinality_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | smote | smote | 5 | 0.5813 | 0.0008 | 0.0456 | 0.8978 | 0.5834 | 2.1041 |
| Logistic Regression | oversampling | oversampling | 5 | 0.5789 | 0.0009 | 0.0432 | 0.9009 | 0.5867 | 2.1767 |
| Logistic Regression | class_weight | none | 5 | 0.5787 | 0.0000 | 0.0430 | 0.9018 | 0.5884 | 2.1845 |
| Logistic Regression | undersampling | undersampling | 5 | 0.5660 | 0.0018 | 0.0303 | 0.8943 | 0.5676 | 2.2655 |
| Logistic Regression | unbalanced | none | 5 | 0.5357 | 0.0000 | 0.0000 | 0.9006 | 0.5957 | 1.5722 |
| XGBoost | scale_pos_weight | none | 5 | 0.5953 | 0.0018 | 0.0658 | 0.8994 | 0.5982 | 1.8004 |
| XGBoost | oversampling | oversampling | 5 | 0.5921 | 0.0027 | 0.0626 | 0.8994 | 0.5985 | 1.7991 |
| XGBoost | smote | smote | 5 | 0.5912 | 0.0033 | 0.0617 | 0.8984 | 0.5955 | 1.8528 |
| XGBoost | undersampling | undersampling | 5 | 0.5773 | 0.0016 | 0.0478 | 0.9001 | 0.5909 | 2.2551 |
| XGBoost | unbalanced | none | 5 | 0.5295 | 0.0012 | 0.0000 | 0.9015 | 0.5983 | 1.5739 |

## Per-Label Metrics

| model_family | strategy | label | positive_rate | f1_mean | delta_f1_mean_vs_unbalanced | pr_auc_mean | delta_pr_auc_mean_vs_unbalanced | predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | class_weight | anticipation | 0.6020 | 0.7546 | -0.0539 | 0.7776 | -0.0001 | 0.6178 |
| Logistic Regression | oversampling | anticipation | 0.6020 | 0.7565 | -0.0520 | 0.7778 | 0.0001 | 0.6198 |
| Logistic Regression | smote | anticipation | 0.6020 | 0.7552 | -0.0533 | 0.7784 | 0.0007 | 0.6123 |
| Logistic Regression | unbalanced | anticipation | 0.6020 | 0.8085 | 0.0000 | 0.7777 | 0.0000 | 0.7568 |
| Logistic Regression | undersampling | anticipation | 0.6020 | 0.7531 | -0.0554 | 0.7763 | -0.0014 | 0.6135 |
| Logistic Regression | class_weight | disgust | 0.0251 | 0.2329 | 0.1878 | 0.1516 | -0.0077 | 0.1520 |
| Logistic Regression | oversampling | disgust | 0.0251 | 0.2337 | 0.1886 | 0.1491 | -0.0102 | 0.1496 |
| Logistic Regression | smote | disgust | 0.0251 | 0.2432 | 0.1981 | 0.1467 | -0.0126 | 0.1291 |
| Logistic Regression | unbalanced | disgust | 0.0251 | 0.0451 | 0.0000 | 0.1593 | 0.0000 | 0.0032 |
| Logistic Regression | undersampling | disgust | 0.0251 | 0.2008 | 0.1557 | 0.1285 | -0.0309 | 0.1950 |
| Logistic Regression | class_weight | fear | 0.1135 | 0.7064 | 0.0006 | 0.7108 | -0.0009 | 0.1898 |
| Logistic Regression | oversampling | fear | 0.1135 | 0.7076 | 0.0018 | 0.7065 | -0.0053 | 0.1891 |
| Logistic Regression | smote | fear | 0.1135 | 0.7148 | 0.0090 | 0.7000 | -0.0118 | 0.1838 |
| Logistic Regression | unbalanced | fear | 0.1135 | 0.7058 | 0.0000 | 0.7117 | 0.0000 | 0.1220 |
| Logistic Regression | undersampling | fear | 0.1135 | 0.6941 | -0.0117 | 0.6772 | -0.0346 | 0.1971 |
| Logistic Regression | class_weight | joy | 0.0923 | 0.4241 | -0.1390 | 0.4944 | -0.0193 | 0.2426 |
| Logistic Regression | oversampling | joy | 0.0923 | 0.4253 | -0.1378 | 0.4930 | -0.0207 | 0.2406 |
| Logistic Regression | smote | joy | 0.0923 | 0.4312 | -0.1319 | 0.4662 | -0.0475 | 0.2286 |
| Logistic Regression | unbalanced | joy | 0.0923 | 0.5631 | 0.0000 | 0.5137 | 0.0000 | 0.0610 |
| Logistic Regression | undersampling | joy | 0.0923 | 0.4096 | -0.1535 | 0.4439 | -0.0699 | 0.2604 |
| Logistic Regression | class_weight | sadness | 0.0442 | 0.3694 | 0.2654 | 0.2632 | -0.0171 | 0.1756 |
| Logistic Regression | oversampling | sadness | 0.0442 | 0.3679 | 0.2639 | 0.2613 | -0.0190 | 0.1728 |
| Logistic Regression | smote | sadness | 0.0442 | 0.3708 | 0.2668 | 0.2746 | -0.0057 | 0.1583 |
| Logistic Regression | unbalanced | sadness | 0.0442 | 0.1040 | 0.0000 | 0.2802 | 0.0000 | 0.0089 |
| Logistic Regression | undersampling | sadness | 0.0442 | 0.3447 | 0.2407 | 0.2340 | -0.0462 | 0.1916 |
| Logistic Regression | class_weight | surprise | 0.3051 | 0.7867 | 0.0373 | 0.8393 | -0.0051 | 0.4154 |
| Logistic Regression | oversampling | surprise | 0.3051 | 0.7861 | 0.0367 | 0.8374 | -0.0069 | 0.4133 |
| Logistic Regression | smote | surprise | 0.3051 | 0.7795 | 0.0301 | 0.8358 | -0.0085 | 0.4066 |
| Logistic Regression | unbalanced | surprise | 0.3051 | 0.7494 | 0.0000 | 0.8443 | 0.0000 | 0.3151 |
| Logistic Regression | undersampling | surprise | 0.3051 | 0.7826 | 0.0332 | 0.8333 | -0.0110 | 0.4135 |
| Logistic Regression | class_weight | trust | 0.3506 | 0.7771 | 0.0031 | 0.8822 | -0.0010 | 0.3914 |
| Logistic Regression | oversampling | trust | 0.3506 | 0.7751 | 0.0011 | 0.8815 | -0.0017 | 0.3914 |
| Logistic Regression | smote | trust | 0.3506 | 0.7748 | 0.0008 | 0.8820 | -0.0012 | 0.3853 |
| Logistic Regression | unbalanced | trust | 0.3506 | 0.7740 | 0.0000 | 0.8832 | 0.0000 | 0.3052 |
| Logistic Regression | undersampling | trust | 0.3506 | 0.7770 | 0.0030 | 0.8799 | -0.0032 | 0.3944 |
| XGBoost | oversampling | anticipation | 0.6020 | 0.7826 | -0.0259 | 0.7693 | -0.0026 | 0.6921 |
| XGBoost | scale_pos_weight | anticipation | 0.6020 | 0.7790 | -0.0296 | 0.7713 | -0.0006 | 0.6785 |
| XGBoost | smote | anticipation | 0.6020 | 0.7731 | -0.0354 | 0.7657 | -0.0063 | 0.6703 |
| XGBoost | unbalanced | anticipation | 0.6020 | 0.8085 | 0.0000 | 0.7719 | 0.0000 | 0.7520 |
| XGBoost | undersampling | anticipation | 0.6020 | 0.7570 | -0.0515 | 0.7694 | -0.0026 | 0.6372 |
| XGBoost | oversampling | disgust | 0.0251 | 0.2134 | 0.1806 | 0.1759 | 0.0011 | 0.0299 |
| XGBoost | scale_pos_weight | disgust | 0.0251 | 0.2311 | 0.1984 | 0.1790 | 0.0043 | 0.0312 |
| XGBoost | smote | disgust | 0.0251 | 0.2437 | 0.2110 | 0.1760 | 0.0012 | 0.0503 |
| XGBoost | unbalanced | disgust | 0.0251 | 0.0327 | 0.0000 | 0.1748 | 0.0000 | 0.0036 |
| XGBoost | undersampling | disgust | 0.0251 | 0.2317 | 0.1990 | 0.1861 | 0.0113 | 0.1728 |
| XGBoost | oversampling | fear | 0.1135 | 0.7251 | 0.0398 | 0.6864 | 0.0052 | 0.1523 |
| XGBoost | scale_pos_weight | fear | 0.1135 | 0.7329 | 0.0476 | 0.6844 | 0.0033 | 0.1520 |
| XGBoost | smote | fear | 0.1135 | 0.7276 | 0.0422 | 0.6854 | 0.0042 | 0.1588 |
| XGBoost | unbalanced | fear | 0.1135 | 0.6853 | 0.0000 | 0.6811 | 0.0000 | 0.1185 |
| XGBoost | undersampling | fear | 0.1135 | 0.6976 | 0.0122 | 0.6626 | -0.0185 | 0.1949 |
| XGBoost | oversampling | joy | 0.0923 | 0.5366 | 0.0034 | 0.5348 | 0.0146 | 0.1085 |
| XGBoost | scale_pos_weight | joy | 0.0923 | 0.5326 | -0.0007 | 0.5327 | 0.0125 | 0.1111 |
| XGBoost | smote | joy | 0.0923 | 0.4922 | -0.0411 | 0.5242 | 0.0040 | 0.1343 |
| XGBoost | unbalanced | joy | 0.0923 | 0.5333 | 0.0000 | 0.5202 | 0.0000 | 0.0565 |
| XGBoost | undersampling | joy | 0.0923 | 0.4205 | -0.1128 | 0.5113 | -0.0089 | 0.2528 |
| XGBoost | oversampling | sadness | 0.0442 | 0.3250 | 0.2117 | 0.2715 | -0.0159 | 0.0743 |
| XGBoost | scale_pos_weight | sadness | 0.0442 | 0.3272 | 0.2138 | 0.2681 | -0.0193 | 0.0725 |
| XGBoost | smote | sadness | 0.0442 | 0.3505 | 0.2372 | 0.2708 | -0.0166 | 0.0875 |
| XGBoost | unbalanced | sadness | 0.0442 | 0.1133 | 0.0000 | 0.2874 | 0.0000 | 0.0098 |
| XGBoost | undersampling | sadness | 0.0442 | 0.3633 | 0.2499 | 0.2606 | -0.0268 | 0.1812 |
| XGBoost | oversampling | surprise | 0.3051 | 0.7750 | 0.0248 | 0.8573 | -0.0012 | 0.3842 |
| XGBoost | scale_pos_weight | surprise | 0.3051 | 0.7772 | 0.0269 | 0.8571 | -0.0013 | 0.3920 |
| XGBoost | smote | surprise | 0.3051 | 0.7676 | 0.0174 | 0.8543 | -0.0041 | 0.3896 |
| XGBoost | unbalanced | surprise | 0.3051 | 0.7503 | 0.0000 | 0.8584 | 0.0000 | 0.3218 |
| XGBoost | undersampling | surprise | 0.3051 | 0.7871 | 0.0369 | 0.8555 | -0.0029 | 0.4234 |
| XGBoost | oversampling | trust | 0.3506 | 0.7870 | 0.0041 | 0.8943 | 0.0003 | 0.3577 |
| XGBoost | scale_pos_weight | trust | 0.3506 | 0.7872 | 0.0043 | 0.8945 | 0.0006 | 0.3630 |
| XGBoost | smote | trust | 0.3506 | 0.7836 | 0.0007 | 0.8921 | -0.0019 | 0.3621 |
| XGBoost | unbalanced | trust | 0.3506 | 0.7829 | 0.0000 | 0.8940 | 0.0000 | 0.3116 |
| XGBoost | undersampling | trust | 0.3506 | 0.7839 | 0.0009 | 0.8909 | -0.0031 | 0.3928 |

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
