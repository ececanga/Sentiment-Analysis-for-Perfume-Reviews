# XGBoost Experiments After Source-Leakage Repair

## Scope

These experiments repeat the XGBoost review-count, fixed-threshold, and balancing analyses after the source-leakage repair. All held-out YouTube evaluation uses the same five-fold source-component manifest:

- manifest: `outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/fold_assignments.csv`
- manifest SHA-256: `5338fbb3f05a51d2dc7ec42564f5cf9d31f3c742dc278e31fb1f4dbf5f47e579`
- YouTube evaluation rows: 784 perfumes
- source groups: 359
- fold sizes: 334, 113, 113, 112, and 112 perfumes
- cross-fold source groups: 0
- seed: 42
- official reporting threshold: 0.3

When synthetic reviews are added, every synthetic row belonging to a held-out perfume ID is excluded from training. The balancing comparison includes only the agreed methods: no balancing, hybrid balancing, and SMOTE.

## 1. Synthetic Review-Count Sweep

Synthetic reviews are added in complete three-review perfume blocks. Metrics are computed only on held-out YouTube perfumes.

| Synthetic reviews | Synthetic perfumes | Macro-F1 | Macro ROC-AUC | Macro PR-AUC | F1 change vs YouTube-only |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.3260 | 0.5683 | 0.3094 | 0.0000 |
| 99 | 33 | 0.3251 | 0.5945 | 0.3061 | -0.0009 |
| 201 | 67 | 0.3241 | 0.5991 | 0.3084 | -0.0018 |
| 300 | 100 | 0.3217 | 0.5780 | 0.3058 | -0.0042 |
| 501 | 167 | 0.3277 | 0.5941 | 0.3028 | +0.0018 |
| 1,002 | 334 | 0.3226 | 0.5617 | 0.3015 | -0.0033 |
| 2,001 | 667 | 0.3235 | 0.5838 | 0.3020 | -0.0025 |
| 2,352 | 784 | 0.3269 | 0.5890 | 0.3066 | +0.0009 |

Interpretation: review count does not produce a monotonic improvement. The best augmented Macro-F1 is 0.3277 at 501 reviews, only 0.0018 above the repaired YouTube-only baseline, while its PR-AUC is lower. This is not evidence that adding more synthetic reviews materially improves held-out YouTube classification.

Final run wall time: 51.62 seconds.

## 2. Fixed-Threshold Sweep

| Threshold | Macro-F1 | Macro ROC-AUC | Macro PR-AUC |
| ---: | ---: | ---: | ---: |
| 0.2 | 0.3409 | 0.5683 | 0.3094 |
| 0.3 | 0.3260 | 0.5683 | 0.3094 |
| 0.5 | 0.2950 | 0.5683 | 0.3094 |
| 0.7 | 0.2248 | 0.5683 | 0.3094 |

Interpretation: 0.2 gives the highest diagnostic fixed-threshold Macro-F1. The thesis-wide prespecified threshold remains 0.3 for the primary comparison; 0.2 is a sensitivity result, not a post-hoc replacement. ROC-AUC and PR-AUC remain identical because they depend on probability rankings rather than the classification cutoff.

Final run wall time: 48.51 seconds.

## 3. Balancing Comparison

All rows below use threshold 0.3 and held-out YouTube evaluation.

| Training setup | Method | Macro-F1 | Macro ROC-AUC | Macro PR-AUC |
| --- | --- | ---: | ---: | ---: |
| YouTube only | No balancing | 0.3242 | 0.5351 | 0.2979 |
| YouTube only | Hybrid | 0.2230 | 0.5328 | 0.3049 |
| YouTube only | SMOTE | 0.3404 | 0.5668 | 0.3150 |
| YouTube + synthetic 784x3 | No balancing | 0.3303 | 0.5770 | 0.3086 |
| YouTube + synthetic 784x3 | Hybrid | 0.3140 | 0.5669 | 0.3131 |
| YouTube + synthetic 784x3 | SMOTE | 0.3258 | 0.6013 | 0.3059 |
| YouTube + synthetic 784x6 | No balancing | 0.3297 | 0.6051 | 0.3651 |
| YouTube + synthetic 784x6 | Hybrid | **0.3794** | 0.5900 | 0.3531 |
| YouTube + synthetic 784x6 | SMOTE | 0.3517 | 0.5819 | 0.3074 |

Interpretation: hybrid balancing with YouTube+784x6 has the highest Macro-F1, but it does not have the best ranking metrics. The unbalanced YouTube+784x6 condition has the highest ROC-AUC and PR-AUC. SMOTE is strongest for YouTube-only Macro-F1, while hybrid is harmful on YouTube-only and YouTube+784x3. Balancing is therefore setup-dependent, not universally beneficial.

Final run wall time: 78.12 seconds.

## Validation

`validation_report.json` independently verifies:

- exactly 784 unique held-out perfume predictions per condition;
- exact agreement between every prediction fold and the repaired manifest;
- zero source-group crossings;
- no missing probabilities;
- exact agreement between stored predictions and each fixed threshold;
- recomputed Macro-F1, Macro ROC-AUC, and Macro PR-AUC matching every recorded result;
- identical XGBoost probability rankings across the four fixed-threshold conditions.

The final artifact tree is protected by `SHA256SUMS`.
