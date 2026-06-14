# 100 Review-Level Pilot Comparison

Both datasets use 100 review rows, Gemini 2.5 Pro labels, and the same LR feature families.

## Metric Summary

- synthetic_v2_100 (all_evaluable): Macro F1=0.6503, ROC AUC=0.8301, PR AUC=0.6768, label cardinality=1.69
- synthetic_v2_100 (common_6_labels): Macro F1=0.6503, ROC AUC=0.8301, PR AUC=0.6768, label cardinality=1.69
- youtube_100 (all_evaluable): Macro F1=0.4514, ROC AUC=0.6791, PR AUC=0.4256, label cardinality=2.11
- youtube_100 (common_6_labels): Macro F1=0.4954, ROC AUC=0.6513, PR AUC=0.4801, label cardinality=2.02

## Label Distribution

| label | synthetic_v2_100 | youtube_100 |
| --- | ---: | ---: |
| anger | 0 | 2 |
| anticipation | 43 | 42 |
| disgust | 0 | 7 |
| fear | 14 | 2 |
| joy | 28 | 77 |
| sadness | 5 | 9 |
| surprise | 44 | 9 |
| trust | 35 | 63 |

## Notes

- The synthetic pilot is exactly 20 perfumes x 5 reviews.
- The YouTube pilot is 100 real review rows: 90 from the same 20 perfumes and 10 filler rows from the next available perfumes because the same 20 had only 90 available real rows without duplication.
- Synthetic has no positive anger or disgust labels in this pilot, so the common six-label comparison is the cleaner model-score comparison.
