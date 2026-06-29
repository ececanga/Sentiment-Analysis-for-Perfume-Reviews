# Randomized Synthetic-Only Review Count Sweep

Synthetic-only grouped cross-validation with random perfume-order subsets. Each count is a complete 6-review perfume block.

Permutation seeds: 101, 202, 303, 404, 505
Thresholds: 0.3

Figures:

- `figures/synthetic_only_random_review_count_sweep_metrics.svg`
- `figures/synthetic_only_random_review_count_sweep_metrics.png`
- `figures/synthetic_only_random_review_count_sweep_table.svg`
- `figures/synthetic_only_random_review_count_sweep_table.png`

| model_family | threshold | synthetic_reviews | synthetic_perfumes | runs | macro_f1_mean | macro_f1_std | macro_roc_auc_mean | macro_roc_auc_std | macro_pr_auc_mean | macro_pr_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.3000 | 198 | 33 | 5 | 0.5524 | 0.0622 | 0.8084 | 0.0351 | 0.5496 | 0.0635 |
| Logistic Regression | 0.3000 | 402 | 67 | 5 | 0.5556 | 0.0264 | 0.8374 | 0.0251 | 0.5489 | 0.0338 |
| Logistic Regression | 0.3000 | 600 | 100 | 5 | 0.5744 | 0.0167 | 0.8727 | 0.0115 | 0.5624 | 0.0284 |
| Logistic Regression | 0.3000 | 1002 | 167 | 5 | 0.5790 | 0.0112 | 0.8802 | 0.0029 | 0.5737 | 0.0180 |
| Logistic Regression | 0.3000 | 2004 | 334 | 5 | 0.5874 | 0.0079 | 0.8932 | 0.0046 | 0.5853 | 0.0042 |
| Logistic Regression | 0.3000 | 4002 | 667 | 5 | 0.5829 | 0.0054 | 0.8987 | 0.0009 | 0.5930 | 0.0031 |
| Logistic Regression | 0.3000 | 4704 | 784 | 5 | 0.5876 | 0.0000 | 0.9006 | 0.0000 | 0.5957 | 0.0000 |
| XGBoost | 0.3000 | 198 | 33 | 5 | 0.5495 | 0.0675 | 0.8293 | 0.0406 | 0.6086 | 0.0562 |
| XGBoost | 0.3000 | 402 | 67 | 5 | 0.5494 | 0.0173 | 0.8813 | 0.0130 | 0.5850 | 0.0284 |
| XGBoost | 0.3000 | 600 | 100 | 5 | 0.5638 | 0.0282 | 0.8862 | 0.0153 | 0.6030 | 0.0274 |
| XGBoost | 0.3000 | 1002 | 167 | 5 | 0.5616 | 0.0096 | 0.8934 | 0.0059 | 0.5935 | 0.0208 |
| XGBoost | 0.3000 | 2004 | 334 | 5 | 0.5596 | 0.0124 | 0.8974 | 0.0043 | 0.5941 | 0.0074 |
| XGBoost | 0.3000 | 4002 | 667 | 5 | 0.5797 | 0.0052 | 0.9023 | 0.0006 | 0.5998 | 0.0035 |
| XGBoost | 0.3000 | 4704 | 784 | 5 | 0.5800 | 0.0018 | 0.9015 | 0.0004 | 0.5983 | 0.0006 |
