# Weighted Synthetic-Augmentation XGBoost Comparison

The accepted XGBoost specification was evaluated with the same repaired source-grouped folds under two conditions. A YouTube perfume has training weight 1; its three synthetic reviews have weight 1/3 each. Test rows are held-out YouTube perfumes only.

## Out-of-fold results

| Condition | Macro-F1 | Macro average precision | Macro ROC-AUC |
| --- | ---: | ---: | ---: |
| youtube_only | 0.3260 | 0.3094 | 0.5683 |
| youtube_plus_weighted_synthetic_784x3 | 0.3359 | 0.3108 | 0.5982 |

## Paired source-component differences

Positive values favor weighted synthetic augmentation.

| Label scope | Metric | Difference | 95% interval | Improvement established |
| --- | --- | ---: | ---: | --- |
| seven_labels | Macro-F1 | +0.0092 | [-0.0205, 0.0393] | no |
| seven_labels | Macro average precision | +0.0025 | [-0.0205, 0.0381] | no |
| seven_labels | Macro ROC-AUC | +0.0286 | [-0.0076, 0.0604] | no |
| fear_excluded_six_labels | Macro-F1 | +0.0107 | [-0.0239, 0.0459] | no |
| fear_excluded_six_labels | Macro average precision | +0.0067 | [-0.0155, 0.0432] | no |
| fear_excluded_six_labels | Macro ROC-AUC | +0.0345 | [0.0031, 0.0701] | yes |

All three metrics are reported without selecting one as the sole main result. The fear-excluded analysis is a sensitivity check because only three YouTube perfumes have a positive fear label.

This remains a mixed-granularity sensitivity analysis: weighting equalizes total influence per perfume, but it does not make generated reviews equivalent to natural YouTube observations.
