# Unweighted Engineered-Feature Logistic Regression Sensitivity

The accepted class-weighted LR and this unweighted LR use the same frozen 784-perfume feature table, source-grouped outer folds, shared inner-validation manifest, seed, features, and threshold. The only intended model change is `class_weight="balanced"` to `class_weight=None`.

| Metric | Class-weighted LR | Unweighted LR | Unweighted minus weighted |
| --- | ---: | ---: | ---: |
| Macro-F1 | 0.3592 | 0.3680 | +0.0088 |
| Macro ROC-AUC | 0.5473 | 0.5826 | +0.0354 |
| Macro average precision | 0.3145 | 0.3285 | +0.0140 |

This is a post-hoc sensitivity comparison. It does not replace the prespecified class-weighted LR row in the primary five-model benchmark.
