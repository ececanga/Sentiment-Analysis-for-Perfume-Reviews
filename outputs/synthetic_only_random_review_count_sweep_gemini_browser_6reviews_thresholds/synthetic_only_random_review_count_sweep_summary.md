# Randomized Synthetic-Only Review Count Sweep

Synthetic-only grouped cross-validation with random perfume-order subsets. Each count is a complete 6-review perfume block.

Permutation seeds: 101, 202, 303, 404, 505
Thresholds: 0.25, 0.3, 0.5, 0.7

Figures:

- `figures/synthetic_only_random_review_count_threshold_sweep_f1.svg`
- `figures/synthetic_only_random_review_count_threshold_sweep_f1.png`
- `figures/synthetic_only_random_review_count_threshold_sweep_table.svg`
- `figures/synthetic_only_random_review_count_threshold_sweep_table.png`

Threshold note:

- Threshold changes Macro F1 and other hard-label metrics, but ROC-AUC and PR-AUC are unchanged because they use ranking/probability scores.
- For Logistic Regression on the full 4,704-review condition, a finer exploratory grid from 0.44 to 0.60 found the highest observed Macro F1 at threshold 0.49: 0.579140.
- This is effectively tied with threshold 0.5, which gives Macro F1 0.578742, so 0.5 is the cleaner value to report unless formal nested threshold selection is added.

| model_family | threshold | synthetic_reviews | synthetic_perfumes | runs | macro_f1_mean | macro_f1_std | macro_roc_auc_mean | macro_roc_auc_std | macro_pr_auc_mean | macro_pr_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.2500 | 198 | 33 | 5 | 0.5370 | 0.0502 | 0.8060 | 0.0321 | 0.5521 | 0.0629 |
| Logistic Regression | 0.2500 | 402 | 67 | 5 | 0.5350 | 0.0255 | 0.8362 | 0.0265 | 0.5489 | 0.0349 |
| Logistic Regression | 0.2500 | 600 | 100 | 5 | 0.5495 | 0.0084 | 0.8705 | 0.0109 | 0.5523 | 0.0267 |
| Logistic Regression | 0.2500 | 1002 | 167 | 5 | 0.5580 | 0.0104 | 0.8784 | 0.0033 | 0.5685 | 0.0168 |
| Logistic Regression | 0.2500 | 2004 | 334 | 5 | 0.5576 | 0.0041 | 0.8928 | 0.0040 | 0.5763 | 0.0054 |
| Logistic Regression | 0.2500 | 4002 | 667 | 5 | 0.5530 | 0.0027 | 0.8999 | 0.0011 | 0.5834 | 0.0016 |
| Logistic Regression | 0.2500 | 4704 | 784 | 5 | 0.5545 | 0.0000 | 0.9018 | 0.0000 | 0.5884 | 0.0000 |
| Logistic Regression | 0.3000 | 198 | 33 | 5 | 0.5466 | 0.0498 | 0.8060 | 0.0321 | 0.5521 | 0.0629 |
| Logistic Regression | 0.3000 | 402 | 67 | 5 | 0.5451 | 0.0225 | 0.8362 | 0.0265 | 0.5489 | 0.0349 |
| Logistic Regression | 0.3000 | 600 | 100 | 5 | 0.5570 | 0.0084 | 0.8705 | 0.0109 | 0.5523 | 0.0267 |
| Logistic Regression | 0.3000 | 1002 | 167 | 5 | 0.5654 | 0.0128 | 0.8784 | 0.0033 | 0.5685 | 0.0168 |
| Logistic Regression | 0.3000 | 2004 | 334 | 5 | 0.5656 | 0.0038 | 0.8928 | 0.0040 | 0.5763 | 0.0054 |
| Logistic Regression | 0.3000 | 4002 | 667 | 5 | 0.5621 | 0.0018 | 0.8999 | 0.0011 | 0.5834 | 0.0016 |
| Logistic Regression | 0.3000 | 4704 | 784 | 5 | 0.5640 | 0.0000 | 0.9018 | 0.0000 | 0.5884 | 0.0000 |
| Logistic Regression | 0.5000 | 198 | 33 | 5 | 0.5442 | 0.0610 | 0.8060 | 0.0321 | 0.5521 | 0.0629 |
| Logistic Regression | 0.5000 | 402 | 67 | 5 | 0.5464 | 0.0279 | 0.8362 | 0.0265 | 0.5489 | 0.0349 |
| Logistic Regression | 0.5000 | 600 | 100 | 5 | 0.5673 | 0.0146 | 0.8705 | 0.0109 | 0.5523 | 0.0267 |
| Logistic Regression | 0.5000 | 1002 | 167 | 5 | 0.5730 | 0.0131 | 0.8784 | 0.0033 | 0.5685 | 0.0168 |
| Logistic Regression | 0.5000 | 2004 | 334 | 5 | 0.5795 | 0.0024 | 0.8928 | 0.0040 | 0.5763 | 0.0054 |
| Logistic Regression | 0.5000 | 4002 | 667 | 5 | 0.5780 | 0.0011 | 0.8999 | 0.0011 | 0.5834 | 0.0016 |
| Logistic Regression | 0.5000 | 4704 | 784 | 5 | 0.5787 | 0.0000 | 0.9018 | 0.0000 | 0.5884 | 0.0000 |
| Logistic Regression | 0.7000 | 198 | 33 | 5 | 0.5284 | 0.0646 | 0.8060 | 0.0321 | 0.5521 | 0.0629 |
| Logistic Regression | 0.7000 | 402 | 67 | 5 | 0.5201 | 0.0381 | 0.8362 | 0.0265 | 0.5489 | 0.0349 |
| Logistic Regression | 0.7000 | 600 | 100 | 5 | 0.5407 | 0.0197 | 0.8705 | 0.0109 | 0.5523 | 0.0267 |
| Logistic Regression | 0.7000 | 1002 | 167 | 5 | 0.5475 | 0.0111 | 0.8784 | 0.0033 | 0.5685 | 0.0168 |
| Logistic Regression | 0.7000 | 2004 | 334 | 5 | 0.5546 | 0.0042 | 0.8928 | 0.0040 | 0.5763 | 0.0054 |
| Logistic Regression | 0.7000 | 4002 | 667 | 5 | 0.5587 | 0.0017 | 0.8999 | 0.0011 | 0.5834 | 0.0016 |
| Logistic Regression | 0.7000 | 4704 | 784 | 5 | 0.5578 | 0.0000 | 0.9018 | 0.0000 | 0.5884 | 0.0000 |
| XGBoost | 0.2500 | 198 | 33 | 5 | 0.5794 | 0.0778 | 0.8397 | 0.0433 | 0.6147 | 0.0629 |
| XGBoost | 0.2500 | 402 | 67 | 5 | 0.5780 | 0.0232 | 0.8813 | 0.0130 | 0.5856 | 0.0185 |
| XGBoost | 0.2500 | 600 | 100 | 5 | 0.5967 | 0.0269 | 0.8868 | 0.0119 | 0.6081 | 0.0181 |
| XGBoost | 0.2500 | 1002 | 167 | 5 | 0.5871 | 0.0137 | 0.8884 | 0.0073 | 0.5927 | 0.0214 |
| XGBoost | 0.2500 | 2004 | 334 | 5 | 0.5908 | 0.0108 | 0.8951 | 0.0046 | 0.5920 | 0.0078 |
| XGBoost | 0.2500 | 4002 | 667 | 5 | 0.5972 | 0.0042 | 0.9004 | 0.0012 | 0.5986 | 0.0048 |
| XGBoost | 0.2500 | 4704 | 784 | 5 | 0.5958 | 0.0023 | 0.8994 | 0.0008 | 0.5982 | 0.0013 |
| XGBoost | 0.3000 | 198 | 33 | 5 | 0.5829 | 0.0850 | 0.8397 | 0.0433 | 0.6147 | 0.0629 |
| XGBoost | 0.3000 | 402 | 67 | 5 | 0.5775 | 0.0266 | 0.8813 | 0.0130 | 0.5856 | 0.0185 |
| XGBoost | 0.3000 | 600 | 100 | 5 | 0.5938 | 0.0307 | 0.8868 | 0.0119 | 0.6081 | 0.0181 |
| XGBoost | 0.3000 | 1002 | 167 | 5 | 0.5823 | 0.0113 | 0.8884 | 0.0073 | 0.5927 | 0.0214 |
| XGBoost | 0.3000 | 2004 | 334 | 5 | 0.5895 | 0.0137 | 0.8951 | 0.0046 | 0.5920 | 0.0078 |
| XGBoost | 0.3000 | 4002 | 667 | 5 | 0.5997 | 0.0046 | 0.9004 | 0.0012 | 0.5986 | 0.0048 |
| XGBoost | 0.3000 | 4704 | 784 | 5 | 0.5982 | 0.0021 | 0.8994 | 0.0008 | 0.5982 | 0.0013 |
| XGBoost | 0.5000 | 198 | 33 | 5 | 0.5719 | 0.0902 | 0.8397 | 0.0433 | 0.6147 | 0.0629 |
| XGBoost | 0.5000 | 402 | 67 | 5 | 0.5527 | 0.0417 | 0.8813 | 0.0130 | 0.5856 | 0.0185 |
| XGBoost | 0.5000 | 600 | 100 | 5 | 0.5614 | 0.0266 | 0.8868 | 0.0119 | 0.6081 | 0.0181 |
| XGBoost | 0.5000 | 1002 | 167 | 5 | 0.5650 | 0.0159 | 0.8884 | 0.0073 | 0.5927 | 0.0214 |
| XGBoost | 0.5000 | 2004 | 334 | 5 | 0.5678 | 0.0124 | 0.8951 | 0.0046 | 0.5920 | 0.0078 |
| XGBoost | 0.5000 | 4002 | 667 | 5 | 0.5902 | 0.0042 | 0.9004 | 0.0012 | 0.5986 | 0.0048 |
| XGBoost | 0.5000 | 4704 | 784 | 5 | 0.5953 | 0.0018 | 0.8994 | 0.0008 | 0.5982 | 0.0013 |
| XGBoost | 0.7000 | 198 | 33 | 5 | 0.5266 | 0.0693 | 0.8397 | 0.0433 | 0.6147 | 0.0629 |
| XGBoost | 0.7000 | 402 | 67 | 5 | 0.5155 | 0.0288 | 0.8813 | 0.0130 | 0.5856 | 0.0185 |
| XGBoost | 0.7000 | 600 | 100 | 5 | 0.5334 | 0.0228 | 0.8868 | 0.0119 | 0.6081 | 0.0181 |
| XGBoost | 0.7000 | 1002 | 167 | 5 | 0.5187 | 0.0196 | 0.8884 | 0.0073 | 0.5927 | 0.0214 |
| XGBoost | 0.7000 | 2004 | 334 | 5 | 0.5141 | 0.0099 | 0.8951 | 0.0046 | 0.5920 | 0.0078 |
| XGBoost | 0.7000 | 4002 | 667 | 5 | 0.5367 | 0.0051 | 0.9004 | 0.0012 | 0.5986 | 0.0048 |
| XGBoost | 0.7000 | 4704 | 784 | 5 | 0.5384 | 0.0046 | 0.8994 | 0.0008 | 0.5982 | 0.0013 |
