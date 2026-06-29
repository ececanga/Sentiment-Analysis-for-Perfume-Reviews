# Randomized Synthetic-Only Review Count Sweep

Synthetic-only grouped cross-validation with random perfume-order subsets. Each count is a complete 3-review perfume block.

Permutation seeds: 101, 202, 303, 404, 505
Thresholds: 0.3

Figures:

- `figures/synthetic_only_random_review_count_sweep_metrics.svg`
- `figures/synthetic_only_random_review_count_sweep_metrics.png`
- `figures/synthetic_only_random_review_count_sweep_table.svg`
- `figures/synthetic_only_random_review_count_sweep_table.png`

| model_family | threshold | synthetic_reviews | synthetic_perfumes | runs | macro_f1_mean | macro_f1_std | macro_roc_auc_mean | macro_roc_auc_std | macro_pr_auc_mean | macro_pr_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.3000 | 99 | 33 | 5 | 0.5931 | 0.0491 | 0.8497 | 0.0834 | 0.6099 | 0.0319 |
| Logistic Regression | 0.3000 | 201 | 67 | 5 | 0.5998 | 0.0276 | 0.8336 | 0.0752 | 0.6047 | 0.0316 |
| Logistic Regression | 0.3000 | 300 | 100 | 5 | 0.6043 | 0.0141 | 0.8988 | 0.0204 | 0.6053 | 0.0088 |
| Logistic Regression | 0.3000 | 501 | 167 | 5 | 0.6320 | 0.0389 | 0.9178 | 0.0130 | 0.6262 | 0.0466 |
| Logistic Regression | 0.3000 | 1002 | 334 | 5 | 0.6385 | 0.0142 | 0.9336 | 0.0079 | 0.6471 | 0.0104 |
| Logistic Regression | 0.3000 | 2001 | 667 | 5 | 0.6307 | 0.0072 | 0.9370 | 0.0013 | 0.6541 | 0.0033 |
| Logistic Regression | 0.3000 | 2352 | 784 | 5 | 0.6514 | 0.0000 | 0.9428 | 0.0000 | 0.6598 | 0.0000 |
| XGBoost | 0.3000 | 99 | 33 | 5 | 0.5830 | 0.0906 | 0.8570 | 0.0797 | 0.6482 | 0.0736 |
| XGBoost | 0.3000 | 201 | 67 | 5 | 0.5877 | 0.0533 | 0.8502 | 0.0764 | 0.6355 | 0.0394 |
| XGBoost | 0.3000 | 300 | 100 | 5 | 0.6087 | 0.0202 | 0.8919 | 0.0415 | 0.6540 | 0.0263 |
| XGBoost | 0.3000 | 501 | 167 | 5 | 0.6055 | 0.0140 | 0.9300 | 0.0101 | 0.6640 | 0.0211 |
| XGBoost | 0.3000 | 1002 | 334 | 5 | 0.6251 | 0.0157 | 0.9430 | 0.0035 | 0.6645 | 0.0062 |
| XGBoost | 0.3000 | 2001 | 667 | 5 | 0.6318 | 0.0095 | 0.9478 | 0.0020 | 0.6733 | 0.0065 |
| XGBoost | 0.3000 | 2352 | 784 | 5 | 0.6576 | 0.0056 | 0.9498 | 0.0005 | 0.6822 | 0.0015 |
