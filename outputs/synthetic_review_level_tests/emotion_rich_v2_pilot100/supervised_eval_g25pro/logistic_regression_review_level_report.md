# Review-Level Synthetic V2 Pilot Evaluation

- Rows: 100
- Perfumes: 20
- Model: logistic_regression
- Split: GroupKFold(n_splits=5, group=perfume_id)
- Threshold: 0.3

## Metrics

- Macro F1: 0.6503
- Macro ROC AUC: 0.8301
- Macro PR AUC: 0.6768
- Micro F1: 0.7143
- Hamming loss: 0.1933

## Per Label

| label | support_pos | positive_rate | predicted_positive_rate | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| joy | 28 | 0.2800 | 0.4600 | 0.6757 | 0.8814 | 0.7034 |
| sadness | 5 | 0.0500 | 0.2200 | 0.1481 | 0.5389 | 0.1294 |
| fear | 14 | 0.1400 | 0.2200 | 0.7778 | 0.9751 | 0.7630 |
| trust | 35 | 0.3500 | 0.5000 | 0.5176 | 0.7059 | 0.6631 |
| surprise | 44 | 0.4400 | 0.4800 | 0.9348 | 0.9675 | 0.9003 |
| anticipation | 43 | 0.4300 | 0.4900 | 0.8478 | 0.9119 | 0.9017 |

Note: this is a 100-review pilot with Gemini 2.5 Pro pseudo-labels. The split is grouped by perfume, so results are useful as a sanity check but not a final thesis benchmark.
