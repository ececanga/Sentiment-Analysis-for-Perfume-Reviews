# XGBoost Feature-Combination Ablation

The accepted 58-feature XGBoost control was reproduced before six advisor-requested feature combinations were evaluated with identical source-grouped folds, model settings, labels, and threshold.

| Variant | Features | Macro-F1 | Macro average precision | Macro ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Full 58-feature control | 58 | 0.3260 | 0.3094 | 0.5683 |
| NRC only | 10 | 0.3795 | 0.3512 | 0.5615 |
| VAD only | 3 | 0.3272 | 0.2798 | 0.5273 |
| Transformer only | 45 | 0.3220 | 0.3003 | 0.5540 |
| NRC + VAD | 13 | 0.3797 | 0.3418 | 0.5702 |
| NRC + Transformer | 55 | 0.3300 | 0.3039 | 0.5596 |
| VAD + Transformer | 48 | 0.3309 | 0.2996 | 0.5505 |

Paired intervals resample the same 359 source components for the full control and every comparison. Positive differences favor the full 58-feature representation. The fear-excluded rows are a prespecified sensitivity because fear has three positives.

Bootstrap comparison rows: 36.
