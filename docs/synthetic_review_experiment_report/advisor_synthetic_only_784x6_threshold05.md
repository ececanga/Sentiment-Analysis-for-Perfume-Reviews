# Advisor Response: Synthetic-Only 784x6 Review Count Sweep

Updated: 2026-06-22

## Scope

This is an advisor-response / supplementary analysis, not the main thesis benchmark.

The experiment uses only synthetic reviews. It does not mix YouTube review rows into training or evaluation. The goal is to check whether synthetic-only model performance changes as we increase the number of generated reviews.

## Method

- Corpus: 784 perfumes x 6 synthetic reviews = 4,704 review-level rows.
- Review indices: 1, 2, 3, 4, 5, 6.
- Counts tested: 198, 402, 600, 1002, 2004, 4002, 4704.
- Each count is a complete 6-review perfume block.
- Perfumes were shuffled first using five random seeds: 101, 202, 303, 404, 505.
- Individual reviews were not shuffled away from their perfume.
- Evaluation used 5-fold grouped cross-validation by `perfume_id`.
- Models: Logistic Regression and XGBoost.
- Final reporting threshold: 0.5.
- Metrics: Macro F1, ROC-AUC, PR-AUC.

## Figures

- [Full compact report](table_images/05_synthetic_only_784x6_threshold05_compact_report.png)
- [Balancing ablation report](table_images/06_synthetic_only_balancing_ablation_report.png)
- [Metric plot](table_images/03_synthetic_only_784x6_threshold05_metrics.png)
- [Readable table](table_images/04_synthetic_only_784x6_threshold05_table.png)

Source outputs:

- `outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_threshold05/synthetic_only_random_review_count_sweep_summary.csv`
- `outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_threshold05/synthetic_only_random_review_count_sweep_summary.md`
- `outputs/synthetic_only_balancing_ablation_gemini_browser_6reviews_threshold05/synthetic_only_balancing_ablation_summary.csv`
- `outputs/synthetic_only_balancing_ablation_gemini_browser_6reviews_threshold05/synthetic_only_balancing_ablation_summary.md`
- `outputs/synthetic_only_resampling_ablation_gemini_browser_6reviews_threshold05/synthetic_only_resampling_ablation_summary.csv`
- `outputs/synthetic_only_resampling_ablation_gemini_browser_6reviews_threshold05/synthetic_only_resampling_ablation_summary.md`
- `outputs/synthetic_only_resampling_ablation_gemini_browser_3reviews_threshold05/synthetic_only_resampling_ablation_summary.csv`
- `outputs/synthetic_only_resampling_ablation_gemini_browser_3reviews_threshold05/synthetic_only_resampling_ablation_summary.md`
- `outputs/synthetic_only_resampling_ablation_gemini_browser_3reviews_threshold03/synthetic_only_resampling_ablation_summary.csv`
- `outputs/synthetic_only_resampling_ablation_gemini_browser_3reviews_threshold03/synthetic_only_resampling_ablation_summary.md`
- `outputs/synthetic_only_resampling_ablation_gemini_browser_6reviews_threshold03/synthetic_only_resampling_ablation_summary.csv`
- `outputs/synthetic_only_resampling_ablation_gemini_browser_6reviews_threshold03/synthetic_only_resampling_ablation_summary.md`

## Main Results At Threshold 0.5

| Model | Best Macro F1 count | Best Macro F1 | Full 4,704-review Macro F1 | Full ROC-AUC | Full PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 2,004 | 0.5795 | 0.5787 | 0.9018 | 0.5884 |
| XGBoost | 4,704 | 0.5953 | 0.5953 | 0.8994 | 0.5982 |

## Balancing Ablation

The threshold-0.5 LR/XGBoost sweep already used balanced model settings:

- Logistic Regression: `class_weight="balanced"`.
- XGBoost: label-wise `scale_pos_weight = negative_count / positive_count`.

To isolate whether balancing matters, we compared those balanced settings against deliberately unbalanced versions on the full 4,704-review corpus, again using 5-fold grouped CV by `perfume_id` and five random perfume-order seeds.

| Model | Variant | Macro F1 mean | Macro F1 SD | F1 delta vs unbalanced | ROC-AUC mean | PR-AUC mean |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | unbalanced | 0.5357 | 0.0000 | 0.0000 | 0.9006 | 0.5957 |
| Logistic Regression | balanced | 0.5787 | 0.0000 | +0.0430 | 0.9018 | 0.5884 |
| XGBoost | unbalanced | 0.5295 | 0.0012 | 0.0000 | 0.9015 | 0.5983 |
| XGBoost | balanced | 0.5953 | 0.0018 | +0.0658 | 0.8994 | 0.5982 |

Main interpretation: balancing improves Macro F1 because it makes the models recover rare labels more often, especially sadness and disgust. It is not a universal ranking improvement: PR-AUC is small/mixed because ranking quality is mostly unchanged, and balancing can reduce F1 for some common labels such as anticipation.

## Data-Level Resampling Ablation

We also tested training-fold-only data-level resampling methods. Resampling was applied separately inside each training fold and separately for each binary emotion classifier. Held-out test folds were never resampled.

The resampling methods use `imbalanced-learn` 0.14.2: `RandomOverSampler`, `RandomUnderSampler`, and official `SMOTE`.

| Model | Strategy | Macro F1 mean | Macro F1 SD | F1 delta vs unbalanced | ROC-AUC mean | PR-AUC mean |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | unbalanced | 0.5357 | 0.0000 | 0.0000 | 0.9006 | 0.5957 |
| Logistic Regression | class weight | 0.5787 | 0.0000 | +0.0430 | 0.9018 | 0.5884 |
| Logistic Regression | random oversampling | 0.5789 | 0.0009 | +0.0432 | 0.9009 | 0.5867 |
| Logistic Regression | random undersampling | 0.5660 | 0.0018 | +0.0303 | 0.8943 | 0.5676 |
| Logistic Regression | SMOTE | 0.5813 | 0.0008 | +0.0456 | 0.8978 | 0.5834 |
| XGBoost | unbalanced | 0.5295 | 0.0012 | 0.0000 | 0.9015 | 0.5983 |
| XGBoost | scale_pos_weight | 0.5953 | 0.0018 | +0.0658 | 0.8994 | 0.5982 |
| XGBoost | random oversampling | 0.5921 | 0.0027 | +0.0626 | 0.8994 | 0.5985 |
| XGBoost | random undersampling | 0.5773 | 0.0016 | +0.0478 | 0.9001 | 0.5909 |
| XGBoost | SMOTE | 0.5912 | 0.0033 | +0.0617 | 0.8984 | 0.5955 |

Main interpretation: data-level balancing also improves Macro F1 over the unbalanced baseline. For LR, official SMOTE is slightly best on Macro F1, although random oversampling and class weighting are close. For XGBoost, the existing `scale_pos_weight` setting remains best and is simpler than resampling. Random undersampling is weaker for both models.

## Original 784x3 Resampling Check

We repeated the same official `imbalanced-learn` resampling ablation on the original 784x3 labeled synthetic corpus: 2,352 rows, 784 perfumes, review indices 1, 2, 3, threshold 0.5, 5-fold grouped CV by `perfume_id`, and five perfume-order seeds.

| Model | Strategy | Macro F1 mean | Macro F1 SD | F1 delta vs unbalanced | ROC-AUC mean | PR-AUC mean |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | unbalanced | 0.5915 | 0.0000 | 0.0000 | 0.9428 | 0.6598 |
| Logistic Regression | class weight | 0.6487 | 0.0000 | +0.0572 | 0.9424 | 0.6497 |
| Logistic Regression | random oversampling | 0.6470 | 0.0023 | +0.0555 | 0.9419 | 0.6459 |
| Logistic Regression | random undersampling | 0.6117 | 0.0057 | +0.0201 | 0.9312 | 0.6073 |
| Logistic Regression | SMOTE | 0.6390 | 0.0020 | +0.0475 | 0.9331 | 0.6443 |
| XGBoost | unbalanced | 0.6135 | 0.0042 | 0.0000 | 0.9498 | 0.6822 |
| XGBoost | scale_pos_weight | 0.6585 | 0.0032 | +0.0450 | 0.9483 | 0.6745 |
| XGBoost | random oversampling | 0.6485 | 0.0046 | +0.0350 | 0.9482 | 0.6719 |
| XGBoost | random undersampling | 0.6290 | 0.0050 | +0.0155 | 0.9449 | 0.6649 |
| XGBoost | SMOTE | 0.6544 | 0.0034 | +0.0409 | 0.9470 | 0.6766 |

Main interpretation: the 784x3 check gives the same practical conclusion as the 784x6 ablation. Data-level balancing improves Macro F1 over unbalanced baselines, but the simplest model-level balancing settings remain very competitive. For 784x3, LR `class_weight="balanced"` is slightly best for LR, and XGBoost `scale_pos_weight` remains best for XGBoost. The 784x3 and 784x6 absolute scores should not be read as a clean data-size comparison because the label distributions differ between the corpora.

## Threshold 0.3 Resampling Rerun

We repeated the 784x3 and 784x6 resampling ablations at threshold 0.3. The same official `imbalanced-learn` methods were used, and all resampling still happened only inside each training fold.

Important framing: `unbalanced` is included only as a baseline reference. The balanced-technique comparison is between class weighting or `scale_pos_weight`, random oversampling, random undersampling, and SMOTE.

Balanced-technique winners at threshold 0.3:

| Corpus | Model | Best balanced technique | Macro F1 mean | Balanced-technique order |
|---|---|---|---:|---|
| 784x3 | Logistic Regression | SMOTE | 0.6358 | SMOTE > oversampling > class weight > undersampling |
| 784x3 | XGBoost | scale_pos_weight | 0.6624 | scale_pos_weight > oversampling > SMOTE > undersampling |
| 784x6 | Logistic Regression | SMOTE | 0.5731 | SMOTE > oversampling > class weight > undersampling |
| 784x6 | XGBoost | scale_pos_weight | 0.5982 | scale_pos_weight > SMOTE > oversampling > undersampling |

784x3 results:

| Model | Strategy | Macro F1 mean | Macro F1 SD | F1 delta vs unbalanced | ROC-AUC mean | PR-AUC mean |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | unbalanced | 0.6514 | 0.0000 | 0.0000 | 0.9428 | 0.6598 |
| Logistic Regression | class weight | 0.6277 | 0.0000 | -0.0237 | 0.9424 | 0.6497 |
| Logistic Regression | random oversampling | 0.6319 | 0.0013 | -0.0195 | 0.9419 | 0.6459 |
| Logistic Regression | random undersampling | 0.5946 | 0.0055 | -0.0567 | 0.9312 | 0.6073 |
| Logistic Regression | SMOTE | 0.6358 | 0.0013 | -0.0155 | 0.9331 | 0.6443 |
| XGBoost | unbalanced | 0.6576 | 0.0056 | 0.0000 | 0.9498 | 0.6822 |
| XGBoost | scale_pos_weight | 0.6624 | 0.0046 | +0.0048 | 0.9483 | 0.6745 |
| XGBoost | random oversampling | 0.6581 | 0.0064 | +0.0004 | 0.9482 | 0.6719 |
| XGBoost | random undersampling | 0.6168 | 0.0033 | -0.0408 | 0.9449 | 0.6649 |
| XGBoost | SMOTE | 0.6577 | 0.0020 | +0.0001 | 0.9470 | 0.6766 |

784x6 results:

| Model | Strategy | Macro F1 mean | Macro F1 SD | F1 delta vs unbalanced | ROC-AUC mean | PR-AUC mean |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | unbalanced | 0.5876 | 0.0000 | 0.0000 | 0.9006 | 0.5957 |
| Logistic Regression | class weight | 0.5640 | 0.0000 | -0.0236 | 0.9018 | 0.5884 |
| Logistic Regression | random oversampling | 0.5645 | 0.0008 | -0.0231 | 0.9009 | 0.5867 |
| Logistic Regression | random undersampling | 0.5507 | 0.0014 | -0.0369 | 0.8943 | 0.5676 |
| Logistic Regression | SMOTE | 0.5731 | 0.0007 | -0.0145 | 0.8978 | 0.5834 |
| XGBoost | unbalanced | 0.5800 | 0.0018 | 0.0000 | 0.9015 | 0.5983 |
| XGBoost | scale_pos_weight | 0.5982 | 0.0021 | +0.0182 | 0.8994 | 0.5982 |
| XGBoost | random oversampling | 0.5954 | 0.0026 | +0.0154 | 0.8994 | 0.5985 |
| XGBoost | random undersampling | 0.5706 | 0.0016 | -0.0094 | 0.9001 | 0.5909 |
| XGBoost | SMOTE | 0.5967 | 0.0019 | +0.0167 | 0.8984 | 0.5955 |

Main interpretation for balanced methods: at threshold 0.3, SMOTE is the best balanced technique for Logistic Regression in both 784x3 and 784x6. For XGBoost, `scale_pos_weight` is the best balanced technique in both corpora. Random undersampling is weakest. The unbalanced rows are baseline references, not balancing methods.

## Interpretation

The 0.5 threshold gives the cleanest single-threshold version of the synthetic-only sweep. Logistic Regression improves compared with the earlier 0.3 threshold because 0.5 filters out more weak positive predictions, improving the precision/recall balance for Macro F1.

More synthetic reviews improve stability: the standard deviation is largest at small review counts and becomes very small at high counts. This means the high-count estimates are less sensitive to which perfumes happen to be selected.

The relationship is still not perfectly monotonic. Logistic Regression has its highest mean Macro F1 at 2,004 reviews, with the full 4,704-review condition essentially tied. XGBoost reaches its best Macro F1 at the full 4,704-review condition.

ROC-AUC and PR-AUC are unchanged by the threshold choice because they are ranking/probability metrics. The threshold only changes hard-label metrics such as F1.

## Recommended Wording

> We ran a synthetic-only review-count sweep using the completed 784x6 synthetic corpus. Perfumes were randomly shuffled across five seeds, and each count used complete six-review perfume blocks. After checking threshold sensitivity, we used 0.5 as the single reporting threshold. The results suggest that adding more synthetic reviews improves stability and generally strengthens ranking metrics, but Macro F1 is not perfectly monotonic. This analysis is best treated as advisor-response evidence about the synthetic corpus, not as the main thesis benchmark.
