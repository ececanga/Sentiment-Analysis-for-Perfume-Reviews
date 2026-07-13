# Fair All-Model Fixed-Threshold Comparison

The same saved OOF probability matrix for each model is thresholded at 0.2, 0.3, 0.5, and 0.7. Models are not retrained between thresholds. Threshold 0.3 remains the prespecified official reporting threshold; the other columns are sensitivity diagnostics.

## Macro-F1 by fixed threshold

| model_name | 0.2 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| BERT | 0.3926 | 0.4029 | 0.2899 | 0.0000 |
| BiLSTM + Attention | 0.3515 | 0.3486 | 0.1943 | 0.0000 |
| Logistic Regression (unweighted sensitivity) | 0.3799 | 0.3680 | 0.3484 | 0.2346 |
| Logistic Regression (class-weighted) | 0.3590 | 0.3592 | 0.2985 | 0.1690 |
| Random Forest | 0.3526 | 0.3306 | 0.2952 | 0.1669 |
| TF-IDF + Logistic Regression (text baseline) | 0.3253 | 0.3254 | 0.2999 | 0.1310 |
| XGBoost | 0.3409 | 0.3260 | 0.2950 | 0.2248 |

## Threshold-independent ranking metrics

ROC-AUC and average precision are reported once because changing a cutoff does not change probability rankings.

| model_name | macro_roc_auc | macro_average_precision |
| --- | --- | --- |
| Logistic Regression (class-weighted) | 0.5473 | 0.3145 |
| Logistic Regression (unweighted sensitivity) | 0.5826 | 0.3285 |
| TF-IDF + Logistic Regression (text baseline) | 0.5409 | 0.3095 |
| Random Forest | 0.5638 | 0.3346 |
| XGBoost | 0.5683 | 0.3094 |
| BERT | 0.5533 | 0.3590 |
| BiLSTM + Attention | 0.4507 | 0.2753 |

## Best tested threshold as a diagnostic

| model_name | best_diagnostic_threshold | best_macro_f1 | official_macro_f1 | best_minus_official_macro_f1 |
| --- | --- | --- | --- | --- |
| Logistic Regression (class-weighted) | 0.3000 | 0.3592 | 0.3592 | 0.0000 |
| Logistic Regression (unweighted sensitivity) | 0.2000 | 0.3799 | 0.3680 | 0.0118 |
| TF-IDF + Logistic Regression (text baseline) | 0.3000 | 0.3254 | 0.3254 | 0.0000 |
| Random Forest | 0.2000 | 0.3526 | 0.3306 | 0.0221 |
| XGBoost | 0.2000 | 0.3409 | 0.3260 | 0.0150 |
| BERT | 0.3000 | 0.4029 | 0.4029 | 0.0000 |
| BiLSTM + Attention | 0.2000 | 0.3515 | 0.3486 | 0.0029 |

The unweighted LR is a post-hoc sensitivity model and TF-IDF + LR is a simple text baseline; neither is an additional primary benchmark row.
