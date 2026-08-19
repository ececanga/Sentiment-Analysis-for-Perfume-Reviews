# Thesis Consolidated Results v1

This package consolidates exact values from the frozen source-grouped experiments. It does not retrain models or recompute metrics. All test rows are the same 784 YouTube perfumes; source-grouped uncertainty uses 359 source components; binary predictions use threshold 0.3; confidence intervals use 2,000 component bootstrap repetitions.

## Files

- `youtube_five_model_pooled.csv`: exact accepted pooled metrics for the five YouTube-only models and their input branch.
- `youtube_five_model_component_equal.csv`: seven-label component-equal metrics and 95% intervals for the five YouTube-only models.
- `synthetic_five_model_augmented_pooled.csv`: exact pooled metrics for the five models trained with three weighted synthetic reviews per training perfume.
- `synthetic_five_model_pooled.csv`: pooled YouTube-only, augmented, and augmented-minus-control values for three metrics.
- `synthetic_five_model_component_intervals.csv`: both conditions, both label scopes, three component-equal metrics, and their intervals.
- `synthetic_five_model_component_differences.csv`: paired augmented-minus-control effects and intervals for both label scopes.
- `feature_combination_three_model_pooled.csv`: seven canonical feature combinations for LR, XGBoost, and Random Forest.
- `feature_combination_three_model_component_intervals.csv`: component-equal interval estimates for all structured-model combinations and both scopes.
- `feature_combination_three_model_component_differences.csv`: paired full-minus-comparison effects for six advisor-requested combinations.
- `SOURCE_MANIFEST.csv`, `VALIDATION.json`, `REPORT.json`, and `SHA256SUMS`: provenance and integrity records.

## Five-model pooled results

| Model | YouTube AP | YouTube F1 | YouTube ROC-AUC | Augmented AP | Augmented F1 | Augmented ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.3145 | 0.3592 | 0.5473 | 0.3505 | 0.3737 | 0.5372 |
| XGBoost | 0.3094 | 0.3260 | 0.5683 | 0.3108 | 0.3359 | 0.5982 |
| Random Forest | 0.3346 | 0.3306 | 0.5638 | 0.3145 | 0.3247 | 0.5731 |
| BERT | 0.3590 | 0.4029 | 0.5533 | 0.3210 | 0.3316 | 0.5267 |
| BiLSTM with Attention | 0.2753 | 0.3486 | 0.4507 | 0.2772 | 0.3466 | 0.4490 |

## Paired synthetic effects (primary seven-label scope)

| Model | Metric | Augmented - YouTube | 95% CI lower | 95% CI upper | CI excludes zero |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | component_equal_macro_f1 | 0.0227 | 0.0011 | 0.0466 | True |
| Logistic Regression | component_equal_macro_average_precision | 0.1202 | -0.0366 | 0.1402 | False |
| Logistic Regression | component_equal_macro_roc_auc | 0.0081 | -0.0195 | 0.0358 | False |
| XGBoost | component_equal_macro_f1 | 0.0092 | -0.0205 | 0.0393 | False |
| XGBoost | component_equal_macro_average_precision | 0.0025 | -0.0205 | 0.0381 | False |
| XGBoost | component_equal_macro_roc_auc | 0.0286 | -0.0076 | 0.0604 | False |
| Random Forest | component_equal_macro_f1 | -0.0117 | -0.0378 | 0.0034 | False |
| Random Forest | component_equal_macro_average_precision | -0.0244 | -0.1079 | 0.0154 | False |
| Random Forest | component_equal_macro_roc_auc | 0.0081 | -0.0390 | 0.0564 | False |
| BERT | component_equal_macro_f1 | -0.1417 | -0.1739 | 0.0261 | False |
| BERT | component_equal_macro_average_precision | -0.1458 | -0.1726 | 0.0722 | False |
| BERT | component_equal_macro_roc_auc | -0.0704 | -0.1387 | 0.0567 | False |
| BiLSTM with Attention | component_equal_macro_f1 | -0.0003 | -0.0195 | 0.0203 | False |
| BiLSTM with Attention | component_equal_macro_average_precision | -0.0044 | -0.0434 | 0.0274 | False |
| BiLSTM with Attention | component_equal_macro_roc_auc | -0.0560 | -0.1563 | 0.1595 | False |

A positive difference favors weighted synthetic augmentation. An interval that includes zero does not establish a directional difference at the stated 95% level.

## Structured-model feature-combination results

| Model | Feature combination | Macro F1 | Macro AP | Macro ROC-AUC |
| --- | --- | --- | --- | --- |
| Logistic Regression | Full representation | 0.3592 | 0.3145 | 0.5473 |
| Logistic Regression | NRC only | 0.3591 | 0.3216 | 0.5074 |
| Logistic Regression | VAD only | 0.3544 | 0.3241 | 0.5884 |
| Logistic Regression | Transformer-derived only | 0.3556 | 0.3395 | 0.5772 |
| Logistic Regression | NRC + VAD | 0.3602 | 0.3318 | 0.5338 |
| Logistic Regression | NRC + Transformer-derived | 0.3600 | 0.3130 | 0.5416 |
| Logistic Regression | VAD + Transformer-derived | 0.3544 | 0.3486 | 0.5838 |
| XGBoost | Full representation | 0.3260 | 0.3094 | 0.5683 |
| XGBoost | NRC only | 0.3795 | 0.3512 | 0.5615 |
| XGBoost | VAD only | 0.3272 | 0.2798 | 0.5273 |
| XGBoost | Transformer-derived only | 0.3220 | 0.3003 | 0.5540 |
| XGBoost | NRC + VAD | 0.3797 | 0.3418 | 0.5702 |
| XGBoost | NRC + Transformer-derived | 0.3300 | 0.3039 | 0.5596 |
| XGBoost | VAD + Transformer-derived | 0.3309 | 0.2996 | 0.5505 |
| Random Forest | Full representation | 0.3306 | 0.3346 | 0.5638 |
| Random Forest | NRC only | 0.3202 | 0.2930 | 0.4937 |
| Random Forest | VAD only | 0.3234 | 0.2814 | 0.5220 |
| Random Forest | Transformer-derived only | 0.3248 | 0.3168 | 0.5255 |
| Random Forest | NRC + VAD | 0.3223 | 0.3074 | 0.5374 |
| Random Forest | NRC + Transformer-derived | 0.3303 | 0.3344 | 0.5554 |
| Random Forest | VAD + Transformer-derived | 0.3248 | 0.3126 | 0.5224 |

This feature-combination analysis applies to Logistic Regression, XGBoost, and Random Forest because those models consume the 58-column structured representation. The current BERT and BiLSTM-with-Attention models consume review text directly; applying these NRC/VAD/transformer-derived combinations to them would require new fusion architectures rather than an ablation of the fitted text-only models.

## Interpretation boundary

Gemini labels are model-based reference targets, not human ground truth. Bootstrap intervals quantify source-component sampling uncertainty conditional on the fitted outer-fold models; they do not include retraining uncertainty or multiplicity adjustment.
