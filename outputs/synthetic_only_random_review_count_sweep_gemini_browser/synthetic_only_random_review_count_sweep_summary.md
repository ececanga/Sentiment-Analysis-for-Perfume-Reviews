# Randomized Synthetic-Only Review Count Sweep

Synthetic-only grouped cross-validation with random perfume-order subsets. Each count is a complete 3-review perfume block.

Permutation seeds: 101, 202, 303, 404, 505

Figures:

- `figures/synthetic_only_random_review_count_sweep_metrics.svg`
- `figures/synthetic_only_random_review_count_sweep_metrics.png`
- `figures/synthetic_only_random_review_count_sweep_table.svg`
- `figures/synthetic_only_random_review_count_sweep_table.png`

| model_family | synthetic_reviews | synthetic_perfumes | runs | macro_f1_mean | macro_f1_std | macro_roc_auc_mean | macro_roc_auc_std | macro_pr_auc_mean | macro_pr_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 99 | 33 | 5 | 0.5829 | 0.0375 | 0.8515 | 0.0824 | 0.6161 | 0.0321 |
| Logistic Regression | 201 | 67 | 5 | 0.5711 | 0.0241 | 0.8356 | 0.0748 | 0.6037 | 0.0301 |
| Logistic Regression | 300 | 100 | 5 | 0.5877 | 0.0219 | 0.8969 | 0.0206 | 0.6062 | 0.0101 |
| Logistic Regression | 501 | 167 | 5 | 0.6199 | 0.0271 | 0.9127 | 0.0176 | 0.6265 | 0.0540 |
| Logistic Regression | 1002 | 334 | 5 | 0.6236 | 0.0114 | 0.9324 | 0.0091 | 0.6424 | 0.0108 |
| Logistic Regression | 2001 | 667 | 5 | 0.6242 | 0.0044 | 0.9335 | 0.0022 | 0.6442 | 0.0029 |
| Logistic Regression | 2352 | 784 | 5 | 0.6277 | 0.0000 | 0.9424 | 0.0000 | 0.6497 | 0.0000 |
| XGBoost | 99 | 33 | 5 | 0.5957 | 0.0709 | 0.8571 | 0.0809 | 0.6461 | 0.0673 |
| XGBoost | 201 | 67 | 5 | 0.6105 | 0.0410 | 0.8515 | 0.0777 | 0.6381 | 0.0419 |
| XGBoost | 300 | 100 | 5 | 0.6239 | 0.0190 | 0.9238 | 0.0146 | 0.6596 | 0.0241 |
| XGBoost | 501 | 167 | 5 | 0.6309 | 0.0306 | 0.9277 | 0.0103 | 0.6659 | 0.0250 |
| XGBoost | 1002 | 334 | 5 | 0.6415 | 0.0160 | 0.9413 | 0.0052 | 0.6694 | 0.0108 |
| XGBoost | 2001 | 667 | 5 | 0.6527 | 0.0144 | 0.9464 | 0.0021 | 0.6679 | 0.0046 |
| XGBoost | 2352 | 784 | 5 | 0.6624 | 0.0046 | 0.9483 | 0.0004 | 0.6745 | 0.0017 |
