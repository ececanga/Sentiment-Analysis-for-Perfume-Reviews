# Synthetic-Only Review Count Sweep

Synthetic-only grouped cross-validation. Each count is a complete 3-review perfume block; folds are grouped by perfume.

Small-count caveat: rare labels can have only one class in a subset, so ROC-AUC macro averages use only labels with finite ROC-AUC values.

Plot:

- `figures/synthetic_only_review_count_sweep_metrics.svg`
- `figures/synthetic_only_review_count_sweep_metrics.png`

Rendered table:

- `figures/synthetic_only_review_count_sweep_table.svg`
- `figures/synthetic_only_review_count_sweep_table.png`

| model_family | synthetic_reviews | synthetic_perfumes | macro_f1 | macro_roc_auc | macro_pr_auc | delta_f1_vs_previous_count | delta_roc_auc_vs_previous_count | delta_pr_auc_vs_previous_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 99 | 33 | 0.7119 | 0.9788 | 0.7470 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 201 | 67 | 0.7544 | 0.9756 | 0.7521 | 0.0425 | -0.0032 | 0.0051 |
| Logistic Regression | 300 | 100 | 0.7489 | 0.9782 | 0.7733 | -0.0055 | 0.0027 | 0.0211 |
| Logistic Regression | 501 | 167 | 0.7446 | 0.9828 | 0.8253 | -0.0043 | 0.0045 | 0.0520 |
| Logistic Regression | 1002 | 334 | 0.6953 | 0.9699 | 0.7348 | -0.0492 | -0.0129 | -0.0905 |
| Logistic Regression | 2001 | 667 | 0.6019 | 0.9457 | 0.6362 | -0.0934 | -0.0242 | -0.0985 |
| Logistic Regression | 2352 | 784 | 0.6277 | 0.9424 | 0.6497 | 0.0258 | -0.0033 | 0.0134 |
| XGBoost | 99 | 33 | 0.7397 | 0.9814 | 0.8167 | 0.0000 | 0.0000 | 0.0000 |
| XGBoost | 201 | 67 | 0.7899 | 0.9810 | 0.8188 | 0.0503 | -0.0003 | 0.0021 |
| XGBoost | 300 | 100 | 0.7969 | 0.9869 | 0.8284 | 0.0070 | 0.0059 | 0.0096 |
| XGBoost | 501 | 167 | 0.7956 | 0.9883 | 0.8357 | -0.0013 | 0.0014 | 0.0072 |
| XGBoost | 1002 | 334 | 0.7223 | 0.9780 | 0.7590 | -0.0733 | -0.0103 | -0.0767 |
| XGBoost | 2001 | 667 | 0.6490 | 0.9443 | 0.6441 | -0.0733 | -0.0336 | -0.1149 |
| XGBoost | 2352 | 784 | 0.6641 | 0.9481 | 0.6735 | 0.0151 | 0.0037 | 0.0293 |
