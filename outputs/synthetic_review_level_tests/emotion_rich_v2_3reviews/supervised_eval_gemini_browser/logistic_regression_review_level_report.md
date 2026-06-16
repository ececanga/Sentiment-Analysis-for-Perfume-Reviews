# Review-Level Synthetic V2 Pilot Evaluation

- Rows: 2352
- Perfumes: 784
- Model: logistic_regression
- Split: GroupKFold(n_splits=5, group=perfume_id)
- Threshold: 0.3

## Metrics

- Macro F1: 0.6277
- Macro ROC AUC: 0.9424
- Macro PR AUC: 0.6497
- Micro F1: 0.7807
- Hamming loss: 0.1350

## Per Label

| label | support_pos | positive_rate | predicted_positive_rate | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| joy | 312 | 0.1327 | 0.3746 | 0.4694 | 0.8944 | 0.6086 |
| sadness | 159 | 0.0676 | 0.2020 | 0.4795 | 0.9414 | 0.3996 |
| fear | 221 | 0.0940 | 0.1884 | 0.6386 | 0.9593 | 0.6367 |
| trust | 834 | 0.3546 | 0.4681 | 0.8041 | 0.9447 | 0.9049 |
| surprise | 774 | 0.3291 | 0.4588 | 0.8192 | 0.9414 | 0.8791 |
| anticipation | 1834 | 0.7798 | 0.7878 | 0.9515 | 0.9612 | 0.9876 |
| disgust | 23 | 0.0098 | 0.0599 | 0.2317 | 0.9545 | 0.1313 |

Note: this is a 100-review pilot with Gemini 2.5 Pro pseudo-labels. The split is grouped by perfume, so results are useful as a sanity check but not a final thesis benchmark.
