# Weighted Synthetic-Augmentation BiLSTM + Attention Comparison

The accepted joint seven-label BiLSTM + Attention model was replayed under the frozen source-grouped protocol. The augmented condition adds three synthetic reviews only to each inner-training perfume at base loss weight 1/3 each. Validation and testing contain YouTube text only.

## Out-of-fold results

| Condition | Macro-F1 | Macro average precision | Macro ROC-AUC |
| --- | ---: | ---: | ---: |
| youtube_only | 0.3486 | 0.2753 | 0.4507 |
| youtube_plus_weighted_synthetic_784x3 | 0.3466 | 0.2772 | 0.4490 |

## Paired source-component differences

Positive values favor weighted synthetic augmentation.

| Label scope | Metric | Difference | 95% interval | Improvement established |
| --- | --- | ---: | ---: | --- |
| seven_labels | Macro-F1 | -0.0003 | [-0.0195, 0.0203] | no |
| seven_labels | Macro average precision | -0.0044 | [-0.0434, 0.0274] | no |
| seven_labels | Macro ROC-AUC | -0.0560 | [-0.1563, 0.1595] | no |
| fear_excluded_six_labels | Macro-F1 | +0.0008 | [-0.0212, 0.0249] | no |
| fear_excluded_six_labels | Macro average precision | +0.0073 | [-0.0104, 0.0344] | no |
| fear_excluded_six_labels | Macro ROC-AUC | +0.0846 | [-0.0454, 0.1766] | no |

All three metrics are reported without selecting a single main result. The fear-excluded result is a sensitivity check because only three YouTube perfumes have a positive fear label.

The 1/chunk-count and 1/3 values are base mini-batch loss weights. They equalize corpus-level base mass per perfume but do not imply identical optimizer influence. Synthetic labels are model-assigned reference labels, not human ground truth.
