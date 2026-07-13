# Source-Grouped OOF Confusion and Prediction Diagnostics

All counts use the fixed threshold-0.3 OOF predictions for the same 784 perfumes and repaired source-grouped folds. The five accepted models are supplemented by a post-hoc unweighted-LR sensitivity row and a simple TF-IDF text-baseline row.

## Predicted-positive counts by emotion

| label | actual_positive_count | logistic_regression_weighted | logistic_regression_unweighted | tfidf_logistic_regression | random_forest | xgboost | bert | bilstm_attention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| joy | 622 | 706 | 763 | 784 | 784 | 757 | 784 | 784 |
| sadness | 11 | 127 | 5 | 0 | 0 | 1 | 0 | 526 |
| fear | 3 | 21 | 3 | 0 | 1 | 4 | 1 | 182 |
| trust | 476 | 709 | 752 | 784 | 784 | 683 | 784 | 784 |
| surprise | 44 | 270 | 25 | 0 | 1 | 14 | 2 | 780 |
| anticipation | 366 | 681 | 642 | 782 | 743 | 567 | 784 | 784 |
| disgust | 19 | 151 | 21 | 0 | 0 | 2 | 2 | 489 |

## Overall prediction behavior

| model_name | total_predicted_positive | mean_predicted_labels_per_perfume | total_true_positive | total_false_positive | total_false_negative | micro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression (class-weighted) | 2665 | 3.3992 | 1369 | 1296 | 172 | 0.6510 |
| Logistic Regression (unweighted sensitivity) | 2211 | 2.8202 | 1392 | 819 | 149 | 0.7420 |
| TF-IDF + Logistic Regression (text baseline) | 2350 | 2.9974 | 1464 | 886 | 77 | 0.7525 |
| Random Forest | 2313 | 2.9503 | 1448 | 865 | 93 | 0.7514 |
| XGBoost | 2028 | 2.5867 | 1296 | 732 | 245 | 0.7263 |
| BERT | 2357 | 3.0064 | 1466 | 891 | 75 | 0.7522 |
| BiLSTM + Attention | 4329 | 5.5217 | 1521 | 2808 | 20 | 0.5182 |

The detailed CSV contains TP, FP, TN, FN, precision, recall, specificity, and F1 for every model-emotion pair.
