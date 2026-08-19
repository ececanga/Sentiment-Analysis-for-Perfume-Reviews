# Weighted Synthetic-Augmentation Random Forest Comparison

The accepted Random Forest specification was evaluated with the same repaired source-grouped folds under two conditions. A YouTube perfume has training base weight 1; its three synthetic reviews have base weight 1/3 each. The accepted balanced-subsample rule then applies tree-specific class weights. Test rows are held-out YouTube perfumes only.

## Out-of-fold results

| Condition | Macro-F1 | Macro average precision | Macro ROC-AUC |
| --- | ---: | ---: | ---: |
| youtube_only | 0.3306 | 0.3346 | 0.5638 |
| youtube_plus_weighted_synthetic_784x3 | 0.3247 | 0.3145 | 0.5731 |

## Paired source-component differences

Positive values favor weighted synthetic augmentation.

| Label scope | Metric | Difference | 95% interval | Improvement established |
| --- | --- | ---: | ---: | --- |
| seven_labels | Macro-F1 | -0.0117 | [-0.0378, 0.0034] | no |
| seven_labels | Macro average precision | -0.0244 | [-0.1079, 0.0154] | no |
| seven_labels | Macro ROC-AUC | +0.0081 | [-0.0390, 0.0564] | no |
| fear_excluded_six_labels | Macro-F1 | -0.0137 | [-0.0441, 0.0040] | no |
| fear_excluded_six_labels | Macro average precision | -0.0195 | [-0.0464, 0.0119] | no |
| fear_excluded_six_labels | Macro ROC-AUC | +0.0098 | [-0.0415, 0.0673] | no |

All three metrics are reported without selecting one as the sole main result. The fear-excluded analysis is a sensitivity check because only three YouTube perfumes have a positive fear label.

This remains a mixed-granularity sensitivity analysis. The three generated rows sum to base weight one per perfume before tree-specific class balancing; this does not make generated reviews equivalent to natural YouTube observations or guarantee identical final influence.
