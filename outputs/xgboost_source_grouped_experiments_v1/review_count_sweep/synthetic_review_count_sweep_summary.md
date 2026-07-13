# Synthetic Review Count Sweep

Held-out YouTube evaluation with fixed perfume-level folds. Synthetic rows are added as complete 3-review perfume blocks.

| model_family | synthetic_reviews | synthetic_perfumes | macro_f1 | macro_roc_auc | macro_pr_auc | delta_f1_vs_youtube | delta_roc_auc_vs_youtube | delta_pr_auc_vs_youtube |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0 | 0 | 0.3260 | 0.5683 | 0.3094 | 0.0000 | 0.0000 | 0.0000 |
| XGBoost | 99 | 33 | 0.3251 | 0.5945 | 0.3061 | -0.0009 | 0.0262 | -0.0034 |
| XGBoost | 201 | 67 | 0.3241 | 0.5991 | 0.3084 | -0.0018 | 0.0308 | -0.0011 |
| XGBoost | 300 | 100 | 0.3217 | 0.5780 | 0.3058 | -0.0042 | 0.0097 | -0.0036 |
| XGBoost | 501 | 167 | 0.3277 | 0.5941 | 0.3028 | 0.0018 | 0.0258 | -0.0066 |
| XGBoost | 1002 | 334 | 0.3226 | 0.5617 | 0.3015 | -0.0033 | -0.0066 | -0.0079 |
| XGBoost | 2001 | 667 | 0.3235 | 0.5838 | 0.3020 | -0.0025 | 0.0155 | -0.0074 |
| XGBoost | 2352 | 784 | 0.3269 | 0.5890 | 0.3066 | 0.0009 | 0.0207 | -0.0028 |
