# Advisor Response: OG YouTube + Synthetic Balancing Ablation

Updated: 2026-06-23

## Scope

This repeats the balancing-method comparison after adding OG YouTube rows into training. The main result is **held-out OG YouTube evaluation**, using the frozen OG fold assignments. Mixed all-row metrics are saved as a diagnostic only, because synthetic rows dominate the mixed row counts.

Source outputs:

- `outputs/og_synthetic_balancing_ablation_gemini_browser_thresholds/og_synthetic_balancing_ablation_summary.csv`
- `outputs/og_synthetic_balancing_ablation_gemini_browser_thresholds/og_synthetic_balancing_ablation_balanced_winners.csv`
- `outputs/og_synthetic_balancing_ablation_gemini_browser_thresholds/og_synthetic_balancing_ablation_per_label.csv`
- `outputs/og_synthetic_balancing_ablation_gemini_browser_thresholds/og_synthetic_balancing_ablation_summary.md`
- `outputs/og_synthetic_balancing_ablation_gemini_browser_thresholds/og_synthetic_balancing_ablation_report.json`

## Method

- Data setups: OG only, OG + synthetic 784x3, OG + synthetic 784x6.
- Models: Logistic Regression and XGBoost.
- Thresholds: 0.3 and 0.5.
- Balanced techniques: class weighting or `scale_pos_weight`, random oversampling, random undersampling, and SMOTE.
- Baseline: unbalanced models, included only as a reference.
- Resampling package: `imbalanced-learn` 0.14.2.
- Feature set: 58 common numeric features available across OG, 784x3, and 784x6 files.
- Fold rule: for each OG fold, every row from the held-out perfume IDs is excluded from training, including synthetic rows for those perfumes.

## Balanced-Technique Winners

Held-out OG YouTube rows:

| Threshold | Data setup | Model | Best balanced technique | Macro F1 | ROC-AUC | PR-AUC | F1 delta vs unbalanced |
|---:|---|---|---|---:|---:|---:|---:|
| 0.3 | OG only | Logistic Regression | SMOTE | 0.3812 | 0.6066 | 0.3295 | -0.0152 |
| 0.3 | OG only | XGBoost | scale_pos_weight | 0.3793 | 0.6224 | 0.3470 | +0.0192 |
| 0.3 | OG + synthetic 784x3 | Logistic Regression | SMOTE | 0.3787 | 0.5913 | 0.3265 | -0.0344 |
| 0.3 | OG + synthetic 784x3 | XGBoost | undersampling | 0.4009 | 0.6338 | 0.3515 | +0.0399 |
| 0.3 | OG + synthetic 784x6 | Logistic Regression | SMOTE | 0.4051 | 0.5866 | 0.3191 | +0.0524 |
| 0.3 | OG + synthetic 784x6 | XGBoost | undersampling | 0.3892 | 0.6358 | 0.3498 | -0.0278 |
| 0.5 | OG only | Logistic Regression | SMOTE | 0.3392 | 0.6066 | 0.3295 | -0.0175 |
| 0.5 | OG only | XGBoost | SMOTE | 0.3447 | 0.6378 | 0.3484 | +0.0431 |
| 0.5 | OG + synthetic 784x3 | Logistic Regression | SMOTE | 0.3426 | 0.5913 | 0.3265 | +0.0387 |
| 0.5 | OG + synthetic 784x3 | XGBoost | oversampling | 0.3411 | 0.5938 | 0.3512 | +0.0382 |
| 0.5 | OG + synthetic 784x6 | Logistic Regression | oversampling | 0.3694 | 0.5851 | 0.3279 | +0.0696 |
| 0.5 | OG + synthetic 784x6 | XGBoost | undersampling | 0.3353 | 0.6358 | 0.3498 | +0.0450 |

## Interpretation

For the actual OG-evaluation question, the strongest balanced result is Logistic Regression with OG + synthetic 784x6, SMOTE, and threshold 0.3: Macro F1 0.4051. XGBoost’s best mixed-training balanced result is OG + synthetic 784x3 with undersampling at threshold 0.3: Macro F1 0.4009.

The balanced methods do not uniformly improve all models. At threshold 0.3, LR with OG+784x3 is worse after balancing, while LR with OG+784x6 improves. XGBoost often improves with balancing, but the best technique changes by corpus and threshold. Random undersampling can help XGBoost on held-out OG F1, but it also raises predicted label cardinality and should be interpreted cautiously.

Use the held-out OG rows as the main result. The mixed all-row metrics are useful only as diagnostics because they are dominated by synthetic review rows.
