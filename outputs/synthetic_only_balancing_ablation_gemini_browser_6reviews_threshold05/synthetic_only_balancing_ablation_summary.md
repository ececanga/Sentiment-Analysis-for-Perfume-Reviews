# Synthetic-Only 784x6 Balancing Ablation

This compares deliberately unbalanced models against the balanced settings used by the current LR/XGBoost sweeps.

- Rows: 4704
- Unique perfumes: 784
- Review indices: 1, 2, 3, 4, 5, 6
- Rows per perfume: 6
- Threshold: 0.5
- Split: GroupKFold with 5 folds, grouped by perfume_id
- Repeated over five random perfume-order seeds so the full-corpus fold assignment is not tied to one row order

## Metric Summary

| model_family | balance_setting | balance_detail | runs | macro_f1_mean | macro_f1_std | delta_macro_f1_mean_vs_unbalanced | macro_roc_auc_mean | macro_roc_auc_std | delta_macro_roc_auc_mean_vs_unbalanced | macro_pr_auc_mean | macro_pr_auc_std | delta_macro_pr_auc_mean_vs_unbalanced | predicted_label_cardinality_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | balanced | class_weight="balanced" | 5 | 0.5787 | 0.0000 | 0.0430 | 0.9018 | 0.0000 | 0.0012 | 0.5884 | 0.0000 | -0.0073 | 2.1845 |
| Logistic Regression | unbalanced | class_weight=None | 5 | 0.5357 | 0.0000 | 0.0000 | 0.9006 | 0.0000 | 0.0000 | 0.5957 | 0.0000 | 0.0000 | 1.5722 |
| XGBoost | balanced | scale_pos_weight=labelwise_neg_to_pos_ratio | 5 | 0.5953 | 0.0018 | 0.0658 | 0.8994 | 0.0008 | -0.0021 | 0.5982 | 0.0013 | -0.0001 | 1.8004 |
| XGBoost | unbalanced | scale_pos_weight=1.0 | 5 | 0.5295 | 0.0012 | 0.0000 | 0.9015 | 0.0004 | 0.0000 | 0.5983 | 0.0006 | 0.0000 | 1.5739 |

## Per-Label Metrics

| model_family | balance_setting | label | support_pos | positive_rate | f1_mean | f1_std | delta_f1_mean_vs_unbalanced | pr_auc_mean | pr_auc_std | delta_pr_auc_mean_vs_unbalanced | predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | balanced | anticipation | 2832 | 0.6020 | 0.7546 | 0.0000 | -0.0539 | 0.7776 | 0.0000 | -0.0001 | 0.6178 |
| Logistic Regression | unbalanced | anticipation | 2832 | 0.6020 | 0.8085 | 0.0000 | 0.0000 | 0.7777 | 0.0000 | 0.0000 | 0.7568 |
| Logistic Regression | balanced | disgust | 118 | 0.0251 | 0.2329 | 0.0000 | 0.1878 | 0.1516 | 0.0000 | -0.0077 | 0.1520 |
| Logistic Regression | unbalanced | disgust | 118 | 0.0251 | 0.0451 | 0.0000 | 0.0000 | 0.1593 | 0.0000 | 0.0000 | 0.0032 |
| Logistic Regression | balanced | fear | 534 | 0.1135 | 0.7064 | 0.0000 | 0.0006 | 0.7108 | 0.0000 | -0.0009 | 0.1898 |
| Logistic Regression | unbalanced | fear | 534 | 0.1135 | 0.7058 | 0.0000 | 0.0000 | 0.7117 | 0.0000 | 0.0000 | 0.1220 |
| Logistic Regression | balanced | joy | 434 | 0.0923 | 0.4241 | 0.0000 | -0.1390 | 0.4944 | 0.0000 | -0.0193 | 0.2426 |
| Logistic Regression | unbalanced | joy | 434 | 0.0923 | 0.5631 | 0.0000 | 0.0000 | 0.5137 | 0.0000 | 0.0000 | 0.0610 |
| Logistic Regression | balanced | sadness | 208 | 0.0442 | 0.3694 | 0.0000 | 0.2654 | 0.2632 | 0.0000 | -0.0171 | 0.1756 |
| Logistic Regression | unbalanced | sadness | 208 | 0.0442 | 0.1040 | 0.0000 | 0.0000 | 0.2802 | 0.0000 | 0.0000 | 0.0089 |
| Logistic Regression | balanced | surprise | 1435 | 0.3051 | 0.7867 | 0.0000 | 0.0373 | 0.8393 | 0.0000 | -0.0051 | 0.4154 |
| Logistic Regression | unbalanced | surprise | 1435 | 0.3051 | 0.7494 | 0.0000 | 0.0000 | 0.8443 | 0.0000 | 0.0000 | 0.3151 |
| Logistic Regression | balanced | trust | 1649 | 0.3506 | 0.7771 | 0.0000 | 0.0031 | 0.8822 | 0.0000 | -0.0010 | 0.3914 |
| Logistic Regression | unbalanced | trust | 1649 | 0.3506 | 0.7740 | 0.0002 | 0.0000 | 0.8832 | 0.0000 | 0.0000 | 0.3052 |
| XGBoost | balanced | anticipation | 2832 | 0.6020 | 0.7790 | 0.0010 | -0.0296 | 0.7713 | 0.0010 | -0.0006 | 0.6785 |
| XGBoost | unbalanced | anticipation | 2832 | 0.6020 | 0.8085 | 0.0016 | 0.0000 | 0.7719 | 0.0013 | 0.0000 | 0.7520 |
| XGBoost | balanced | disgust | 118 | 0.0251 | 0.2311 | 0.0182 | 0.1984 | 0.1790 | 0.0041 | 0.0043 | 0.0312 |
| XGBoost | unbalanced | disgust | 118 | 0.0251 | 0.0327 | 0.0073 | 0.0000 | 0.1748 | 0.0038 | 0.0000 | 0.0036 |
| XGBoost | balanced | fear | 534 | 0.1135 | 0.7329 | 0.0019 | 0.0476 | 0.6844 | 0.0029 | 0.0033 | 0.1520 |
| XGBoost | unbalanced | fear | 534 | 0.1135 | 0.6853 | 0.0038 | 0.0000 | 0.6811 | 0.0044 | 0.0000 | 0.1185 |
| XGBoost | balanced | joy | 434 | 0.0923 | 0.5326 | 0.0066 | -0.0007 | 0.5327 | 0.0039 | 0.0125 | 0.1111 |
| XGBoost | unbalanced | joy | 434 | 0.0923 | 0.5333 | 0.0051 | 0.0000 | 0.5202 | 0.0056 | 0.0000 | 0.0565 |
| XGBoost | balanced | sadness | 208 | 0.0442 | 0.3272 | 0.0086 | 0.2138 | 0.2681 | 0.0052 | -0.0193 | 0.0725 |
| XGBoost | unbalanced | sadness | 208 | 0.0442 | 0.1133 | 0.0121 | 0.0000 | 0.2874 | 0.0019 | 0.0000 | 0.0098 |
| XGBoost | balanced | surprise | 1435 | 0.3051 | 0.7772 | 0.0036 | 0.0269 | 0.8571 | 0.0018 | -0.0013 | 0.3920 |
| XGBoost | unbalanced | surprise | 1435 | 0.3051 | 0.7503 | 0.0052 | 0.0000 | 0.8584 | 0.0015 | 0.0000 | 0.3218 |
| XGBoost | balanced | trust | 1649 | 0.3506 | 0.7872 | 0.0024 | 0.0043 | 0.8945 | 0.0010 | 0.0006 | 0.3630 |
| XGBoost | unbalanced | trust | 1649 | 0.3506 | 0.7829 | 0.0033 | 0.0000 | 0.8940 | 0.0007 | 0.0000 | 0.3116 |

## Label Distribution

| label | positive_count | negative_count | positive_rate | neg_to_pos_ratio |
| --- | --- | --- | --- | --- |
| joy | 434 | 4270 | 0.0923 | 9.8387 |
| sadness | 208 | 4496 | 0.0442 | 21.6154 |
| fear | 534 | 4170 | 0.1135 | 7.8090 |
| trust | 1649 | 3055 | 0.3506 | 1.8526 |
| surprise | 1435 | 3269 | 0.3051 | 2.2780 |
| anticipation | 2832 | 1872 | 0.6020 | 0.6610 |
| disgust | 118 | 4586 | 0.0251 | 38.8644 |

## Interpretation

- The earlier threshold-0.5 synthetic-only sweep already used balanced LR and balanced XGBoost.
- This ablation isolates whether that balancing helps compared with deliberately unbalanced versions.
- Balancing should be judged by macro F1 and rare-label F1/PR-AUC, not only by aggregate ROC-AUC.
