# Random Forest Feature-Combination Ablation

The accepted 58-feature Random Forest control was reproduced before six advisor-requested feature combinations were evaluated with identical source-grouped folds, model settings, labels, and threshold.

| Variant | Features | Macro-F1 | Macro average precision | Macro ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Full 58-feature control | 58 | 0.3306 | 0.3346 | 0.5638 |
| NRC only | 10 | 0.3202 | 0.2930 | 0.4937 |
| VAD only | 3 | 0.3234 | 0.2814 | 0.5220 |
| Transformer only | 45 | 0.3248 | 0.3168 | 0.5255 |
| NRC + VAD | 13 | 0.3223 | 0.3074 | 0.5374 |
| NRC + Transformer | 55 | 0.3303 | 0.3344 | 0.5554 |
| VAD + Transformer | 48 | 0.3248 | 0.3126 | 0.5224 |

Paired intervals resample the same 359 source components for the full control and every comparison. Positive differences favor the full 58-feature representation. The fear-excluded rows are a prespecified sensitivity because fear has three positives.

Bootstrap comparison rows: 36.
