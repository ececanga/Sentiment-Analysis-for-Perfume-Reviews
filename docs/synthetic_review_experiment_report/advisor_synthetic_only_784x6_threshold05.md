# Advisor Response: Synthetic-Only 784x6 Review Count Sweep

Updated: 2026-06-20

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
- [Metric plot](table_images/03_synthetic_only_784x6_threshold05_metrics.png)
- [Readable table](table_images/04_synthetic_only_784x6_threshold05_table.png)

Source outputs:

- `outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_threshold05/synthetic_only_random_review_count_sweep_summary.csv`
- `outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_threshold05/synthetic_only_random_review_count_sweep_summary.md`

## Main Results At Threshold 0.5

| Model | Best Macro F1 count | Best Macro F1 | Full 4,704-review Macro F1 | Full ROC-AUC | Full PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 2,004 | 0.5795 | 0.5787 | 0.9018 | 0.5884 |
| XGBoost | 4,704 | 0.5953 | 0.5953 | 0.8994 | 0.5982 |

## Interpretation

The 0.5 threshold gives the cleanest single-threshold version of the synthetic-only sweep. Logistic Regression improves compared with the earlier 0.3 threshold because 0.5 filters out more weak positive predictions, improving the precision/recall balance for Macro F1.

More synthetic reviews improve stability: the standard deviation is largest at small review counts and becomes very small at high counts. This means the high-count estimates are less sensitive to which perfumes happen to be selected.

The relationship is still not perfectly monotonic. Logistic Regression has its highest mean Macro F1 at 2,004 reviews, with the full 4,704-review condition essentially tied. XGBoost reaches its best Macro F1 at the full 4,704-review condition.

ROC-AUC and PR-AUC are unchanged by the threshold choice because they are ranking/probability metrics. The threshold only changes hard-label metrics such as F1.

## Recommended Wording

> We ran a synthetic-only review-count sweep using the completed 784x6 synthetic corpus. Perfumes were randomly shuffled across five seeds, and each count used complete six-review perfume blocks. After checking threshold sensitivity, we used 0.5 as the single reporting threshold. The results suggest that adding more synthetic reviews improves stability and generally strengthens ranking metrics, but Macro F1 is not perfectly monotonic. This analysis is best treated as advisor-response evidence about the synthetic corpus, not as the main thesis benchmark.
