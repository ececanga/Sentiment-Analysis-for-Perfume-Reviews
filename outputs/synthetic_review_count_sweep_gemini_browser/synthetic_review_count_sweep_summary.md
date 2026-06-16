# Synthetic Review Count Sweep

Held-out YouTube evaluation with fixed perfume-level folds. Synthetic rows are added as complete 3-review perfume blocks.

| model_family | synthetic_reviews | synthetic_perfumes | macro_f1 | macro_roc_auc | macro_pr_auc | delta_f1_vs_youtube | delta_roc_auc_vs_youtube | delta_pr_auc_vs_youtube |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0 | 0 | 0.3602 | 0.5630 | 0.3270 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 99 | 33 | 0.3731 | 0.6043 | 0.3508 | 0.0129 | 0.0414 | 0.0238 |
| Logistic Regression | 201 | 67 | 0.3750 | 0.6056 | 0.3595 | 0.0148 | 0.0426 | 0.0325 |
| Logistic Regression | 300 | 100 | 0.3787 | 0.5945 | 0.3400 | 0.0186 | 0.0316 | 0.0131 |
| Logistic Regression | 501 | 167 | 0.3790 | 0.5779 | 0.3320 | 0.0188 | 0.0150 | 0.0050 |
| Logistic Regression | 1002 | 334 | 0.3709 | 0.5791 | 0.3410 | 0.0107 | 0.0161 | 0.0140 |
| Logistic Regression | 2001 | 667 | 0.3705 | 0.5966 | 0.3240 | 0.0103 | 0.0336 | -0.0030 |
| Logistic Regression | 2352 | 784 | 0.3744 | 0.5952 | 0.3312 | 0.0142 | 0.0322 | 0.0043 |
| XGBoost | 0 | 0 | 0.3840 | 0.6218 | 0.3449 | 0.0000 | 0.0000 | 0.0000 |
| XGBoost | 99 | 33 | 0.3706 | 0.6436 | 0.3541 | -0.0133 | 0.0218 | 0.0092 |
| XGBoost | 201 | 67 | 0.3735 | 0.6417 | 0.3502 | -0.0105 | 0.0199 | 0.0053 |
| XGBoost | 300 | 100 | 0.3811 | 0.6383 | 0.3512 | -0.0028 | 0.0165 | 0.0063 |
| XGBoost | 501 | 167 | 0.3617 | 0.6372 | 0.3477 | -0.0223 | 0.0154 | 0.0028 |
| XGBoost | 1002 | 334 | 0.3646 | 0.6388 | 0.3443 | -0.0193 | 0.0169 | -0.0006 |
| XGBoost | 2001 | 667 | 0.3776 | 0.6204 | 0.3483 | -0.0064 | -0.0014 | 0.0034 |
| XGBoost | 2352 | 784 | 0.3725 | 0.6381 | 0.3580 | -0.0115 | 0.0163 | 0.0131 |
