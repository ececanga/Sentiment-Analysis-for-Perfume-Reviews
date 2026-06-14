# Review-Level Synthetic V2 Pilot Evaluation

- Rows: 100
- Perfumes: 30
- Model: logistic_regression
- Split: GroupKFold(n_splits=5, group=perfume_id)
- Threshold: 0.3

## Metrics

- Macro F1: 0.4514
- Macro ROC AUC: 0.6791
- Macro PR AUC: 0.4256
- Micro F1: 0.6047
- Hamming loss: 0.2500

## Per Label

| label | support_pos | positive_rate | predicted_positive_rate | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| joy | 77 | 0.7700 | 0.8300 | 0.8250 | 0.4071 | 0.7029 |
| sadness | 9 | 0.0900 | 0.1700 | 0.3846 | 0.7302 | 0.3294 |
| anger | 2 | 0.0200 | 0.2200 | 0.0833 | 0.6939 | 0.0807 |
| fear | 2 | 0.0200 | 0.1000 | 0.3333 | 0.9694 | 0.3333 |
| trust | 63 | 0.6300 | 0.7300 | 0.6324 | 0.4865 | 0.6286 |
| surprise | 9 | 0.0900 | 0.2600 | 0.2286 | 0.6947 | 0.3506 |
| anticipation | 42 | 0.4200 | 0.5300 | 0.5684 | 0.6201 | 0.5355 |
| disgust | 7 | 0.0700 | 0.1100 | 0.5556 | 0.8310 | 0.4439 |

Note: this is a 100-review pilot with Gemini 2.5 Pro pseudo-labels. The split is grouped by perfume, so results are useful as a sanity check but not a final thesis benchmark.
