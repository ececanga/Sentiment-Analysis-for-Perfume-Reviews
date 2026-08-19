# Weighted Synthetic-Augmentation BERT Comparison

The accepted joint seven-label BERT control is reused without retraining. The augmented condition adds three synthetic reviews only to each inner-training perfume at base loss weight 1/3 each. Validation and testing contain YouTube text only.

## Out-of-fold results

| Condition | Macro-F1 | Macro average precision | Macro ROC-AUC |
| --- | ---: | ---: | ---: |
| youtube_only | 0.4029 | 0.3590 | 0.5533 |
| youtube_plus_weighted_synthetic_784x3 | 0.3316 | 0.3210 | 0.5267 |

## Paired source-component differences

Positive values favor weighted synthetic augmentation.

| Label scope | Metric | Difference | 95% interval | Improvement established |
| --- | --- | ---: | ---: | --- |
| seven_labels | Macro-F1 | -0.1417 | [-0.1739, 0.0261] | no |
| seven_labels | Macro average precision | -0.1458 | [-0.1726, 0.0722] | no |
| seven_labels | Macro ROC-AUC | -0.0704 | [-0.1387, 0.0567] | no |
| fear_excluded_six_labels | Macro-F1 | +0.0004 | [-0.0388, 0.0392] | no |
| fear_excluded_six_labels | Macro average precision | -0.0057 | [-0.0464, 0.0176] | no |
| fear_excluded_six_labels | Macro ROC-AUC | +0.0114 | [-0.0746, 0.0700] | no |

All three metrics are reported without selecting a single main result. The fear-excluded result is a sensitivity check because only three YouTube perfumes have a positive fear label.

The 1/chunk-count and 1/3 values are base mini-batch loss weights. They equalize corpus-level base mass per perfume but do not imply identical optimizer influence. Synthetic labels are model-assigned reference labels, not human ground truth.
