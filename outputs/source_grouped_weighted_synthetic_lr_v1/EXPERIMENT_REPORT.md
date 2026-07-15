# Phase 6: Weighted Synthetic-Augmentation Logistic Regression

This prespecified contrast uses the repaired exact-source-grouped folds. A YouTube perfume contributes training weight 1 and its three synthetic reviews contribute 1/3 each. Evaluation contains only held-out YouTube perfumes.

## Pooled out-of-fold results

| Condition | Macro-F1 | Macro average precision | Macro ROC-AUC |
| --- | ---: | ---: | ---: |
| youtube_only | 0.3592 | 0.3145 | 0.5473 |
| youtube_plus_weighted_synthetic_784x3 | 0.3737 | 0.3505 | 0.5372 |

## Paired component-equal augmentation differences

Positive values favor weighted synthetic augmentation.

| Label scope | Metric | Difference | 95% interval | Excludes zero |
| --- | --- | ---: | ---: | --- |
| seven_labels | Macro-F1 | +0.0227 | [0.0011, 0.0466] | yes |
| seven_labels | Macro average precision | +0.1202 | [-0.0366, 0.1402] | no |
| seven_labels | Macro ROC-AUC | +0.0081 | [-0.0195, 0.0358] | no |
| fear_excluded_six_labels | Macro-F1 | +0.0159 | [-0.0037, 0.0338] | no |
| fear_excluded_six_labels | Macro average precision | -0.0164 | [-0.0516, 0.0099] | no |
| fear_excluded_six_labels | Macro ROC-AUC | +0.0049 | [-0.0273, 0.0367] | no |

## Interpretation boundary

The primary macro-average-precision augmentation difference is +0.1202 with a 95% component interval [-0.0366, 0.1402]. Because this interval includes zero, the primary analysis does not establish an augmentation improvement.
The seven-label component-equal macro-F1 difference is +0.0227 [0.0011, 0.0466], but the fear-excluded difference is +0.0159 [-0.0037, 0.0338].
Fear has only three positive YouTube examples. Its pooled F1 changes from 0.0833 to 0.1429, while its component-equal average precision changes from 0.0480 to 0.9882. After excluding fear, the component-equal macro-average-precision difference is -0.0164 [-0.0516, 0.0099]. The apparent seven-label ranking gain is therefore dominated by an ultra-rare label.
This remains a mixed-granularity sensitivity: real rows are perfume-level aggregates, while synthetic rows are individual generated reviews. Weighting equalizes their total training influence per perfume but does not make their feature or label distributions equivalent.
