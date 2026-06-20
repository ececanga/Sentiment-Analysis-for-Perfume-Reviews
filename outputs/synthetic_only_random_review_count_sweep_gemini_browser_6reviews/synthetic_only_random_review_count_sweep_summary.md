# Randomized Synthetic-Only Review Count Sweep

Synthetic-only grouped cross-validation with random perfume-order subsets. Each count is a complete 6-review perfume block.

Permutation seeds: 101, 202, 303, 404, 505

Figures:

- `figures/synthetic_only_random_review_count_sweep_metrics.svg`
- `figures/synthetic_only_random_review_count_sweep_metrics.png`
- `figures/synthetic_only_random_review_count_sweep_table.svg`
- `figures/synthetic_only_random_review_count_sweep_table.png`

| model_family | synthetic_reviews | synthetic_perfumes | runs | macro_f1_mean | macro_f1_std | macro_roc_auc_mean | macro_roc_auc_std | macro_pr_auc_mean | macro_pr_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 198 | 33 | 5 | 0.5466 | 0.0498 | 0.8060 | 0.0321 | 0.5521 | 0.0629 |
| Logistic Regression | 402 | 67 | 5 | 0.5451 | 0.0225 | 0.8362 | 0.0265 | 0.5489 | 0.0349 |
| Logistic Regression | 600 | 100 | 5 | 0.5570 | 0.0084 | 0.8705 | 0.0109 | 0.5523 | 0.0267 |
| Logistic Regression | 1002 | 167 | 5 | 0.5654 | 0.0128 | 0.8784 | 0.0033 | 0.5685 | 0.0168 |
| Logistic Regression | 2004 | 334 | 5 | 0.5656 | 0.0038 | 0.8928 | 0.0040 | 0.5763 | 0.0054 |
| Logistic Regression | 4002 | 667 | 5 | 0.5621 | 0.0018 | 0.8999 | 0.0011 | 0.5834 | 0.0016 |
| Logistic Regression | 4704 | 784 | 5 | 0.5640 | 0.0000 | 0.9018 | 0.0000 | 0.5884 | 0.0000 |
| XGBoost | 198 | 33 | 5 | 0.5829 | 0.0850 | 0.8397 | 0.0433 | 0.6147 | 0.0629 |
| XGBoost | 402 | 67 | 5 | 0.5775 | 0.0266 | 0.8813 | 0.0130 | 0.5856 | 0.0185 |
| XGBoost | 600 | 100 | 5 | 0.5938 | 0.0307 | 0.8868 | 0.0119 | 0.6081 | 0.0181 |
| XGBoost | 1002 | 167 | 5 | 0.5823 | 0.0113 | 0.8884 | 0.0073 | 0.5927 | 0.0214 |
| XGBoost | 2004 | 334 | 5 | 0.5895 | 0.0137 | 0.8951 | 0.0046 | 0.5920 | 0.0078 |
| XGBoost | 4002 | 667 | 5 | 0.5997 | 0.0046 | 0.9004 | 0.0012 | 0.5986 | 0.0048 |
| XGBoost | 4704 | 784 | 5 | 0.5982 | 0.0021 | 0.8994 | 0.0008 | 0.5982 | 0.0013 |
