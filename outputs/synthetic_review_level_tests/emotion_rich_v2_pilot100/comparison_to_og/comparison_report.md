# Emotion-Rich V2 Pilot vs OG Comparison

## Scope

- Synthetic v2 rows: 100 reviews, 20 perfumes.
- Synthetic v2 reviews were aggregated to perfume level before comparing with OG.
- Shared perfume count: 20.
- Shared numeric feature count: 58.
- Supervised synthetic metrics were not computed because the v2 pilot has no emotion/VAD labels.

## Existing OG Logistic Regression Benchmark

- Rows: 784
- Threshold: 0.3
- Macro F1: 0.3178
- Macro ROC AUC: 0.7294
- Macro PR AUC: 0.3262
- Micro F1: 0.5835

## Synthetic V2 LR Inference

| label | threshold | og_true_positive_rate | synthetic_predicted_positive_rate | synthetic_mean_probability | synthetic_min_probability | synthetic_max_probability | synthetic_positive_count | synthetic_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| joy | 0.3000 | 0.6416 | 0.5000 | 0.3045 | 0.0414 | 0.6927 | 10 | 20 |
| sadness | 0.3000 | 0.0153 | 0.0500 | 0.0543 | 0.0000 | 0.9155 | 1 | 20 |
| fear | 0.3000 | 0.0038 | 0.0000 | 0.0333 | 0.0031 | 0.0788 | 0 | 20 |
| trust | 0.3000 | 0.3010 | 0.6000 | 0.3673 | 0.1109 | 0.7735 | 12 | 20 |
| surprise | 0.3000 | 0.0128 | 0.4500 | 0.3608 | 0.0007 | 1.0000 | 9 | 20 |
| anticipation | 0.3000 | 0.3929 | 0.6500 | 0.3643 | 0.0745 | 0.6791 | 13 | 20 |
| disgust | 0.3000 | 0.0102 | 0.0000 | 0.0004 | 0.0000 | 0.0077 | 0 | 20 |

These are classifier predictions after training on OG labels. They are useful for a pipeline sanity check, but they are not evaluation scores and not ground-truth labels.

## Feature Family Differences

| feature_family | feature_count | compared_values | synthetic_mean | og_mean | mean_delta_synthetic_minus_og | mean_abs_delta | max_abs_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cardiff | 4 | 80 | 0.2500 | 0.2500 | -0.0000 | 0.0815 | 0.4150 |
| GoEmotions | 28 | 560 | 0.0487 | 0.0481 | 0.0006 | 0.0524 | 0.9876 |
| Hartmann | 7 | 140 | 0.1429 | 0.1429 | 0.0000 | 0.1154 | 0.8135 |
| NRC | 10 | 200 | 1.6540 | 137.1000 | -135.4460 | 135.4620 | 1260.8000 |
| Savani | 6 | 120 | 0.1667 | 0.1667 | -0.0000 | 0.1359 | 0.6675 |
| VAD | 3 | 60 | 0.0052 | 0.0307 | -0.0255 | 0.0404 | 0.1198 |

## Largest Mean Feature Differences

| feature_family | feature | synthetic_mean | og_mean | mean_delta_synthetic_minus_og | mean_abs_delta |
| --- | --- | --- | --- | --- | --- |
| NRC | NRC_positive | 4.5600 | 384.7500 | -380.1900 | 380.1900 |
| NRC | NRC_trust | 2.4900 | 240.7000 | -238.2100 | 238.2100 |
| NRC | NRC_joy | 2.2400 | 205.2000 | -202.9600 | 202.9600 |
| NRC | NRC_anticipation | 1.7000 | 133.7500 | -132.0500 | 132.0500 |
| NRC | NRC_negative | 1.4400 | 128.4000 | -126.9600 | 126.9600 |
| NRC | NRC_surprise | 0.8100 | 66.7000 | -65.8900 | 65.8900 |
| NRC | NRC_anger | 0.5500 | 59.2500 | -58.7000 | 58.7200 |
| NRC | NRC_disgust | 0.8200 | 58.0000 | -57.1800 | 57.2600 |
| NRC | NRC_sadness | 0.8300 | 50.2000 | -49.3700 | 49.4300 |
| NRC | NRC_fear | 1.1000 | 44.0500 | -42.9500 | 42.9500 |
| Savani | Savani_joy | 0.5141 | 0.7769 | -0.2629 | 0.3063 |
| Hartmann | Hartmann_neutral | 0.1961 | 0.3949 | -0.1988 | 0.2290 |
| GoEmotions | GoEmotions_neutral | 0.4191 | 0.3641 | 0.0550 | 0.2198 |
| GoEmotions | GoEmotions_realization | 0.2484 | 0.0371 | 0.2113 | 0.2139 |
| Hartmann | Hartmann_fear | 0.2088 | 0.0237 | 0.1851 | 0.1851 |

## NRC Per 100 Words

| feature | synthetic_per_100_words | og_per_100_words | mean_delta_synthetic_minus_og | mean_abs_delta |
| --- | --- | --- | --- | --- |
| NRC_positive | 6.7819 | 5.0460 | 1.7359 | 1.7828 |
| NRC_fear | 1.6342 | 0.5048 | 1.1294 | 1.1294 |
| NRC_trust | 3.6981 | 3.0346 | 0.6635 | 0.8448 |
| NRC_joy | 3.3209 | 2.8245 | 0.4964 | 0.8437 |
| NRC_negative | 2.1321 | 1.6386 | 0.4934 | 0.7605 |
| NRC_anticipation | 2.5275 | 1.8359 | 0.6916 | 0.6916 |
| NRC_sadness | 1.2284 | 0.6202 | 0.6082 | 0.6458 |
| NRC_disgust | 1.2134 | 0.6631 | 0.5503 | 0.6384 |
| NRC_anger | 0.8064 | 0.7659 | 0.0405 | 0.5869 |
| NRC_surprise | 1.2016 | 0.9212 | 0.2804 | 0.4658 |

These normalized values compare lexical emotion density. They are more readable than raw NRC counts, because OG perfume rows aggregate much longer YouTube transcripts.

## Transformer Top Labels

| source | label_column | label | count | share |
| --- | --- | --- | --- | --- |
| synthetic_v2_review_level | JHartmann_Label | joy | 39 | 0.3900 |
| synthetic_v2_review_level | JHartmann_Label | fear | 20 | 0.2000 |
| synthetic_v2_review_level | JHartmann_Label | neutral | 19 | 0.1900 |
| synthetic_v2_review_level | JHartmann_Label | surprise | 9 | 0.0900 |
| synthetic_v2_review_level | JHartmann_Label | disgust | 7 | 0.0700 |
| synthetic_v2_review_level | JHartmann_Label | sadness | 5 | 0.0500 |
| synthetic_v2_review_level | JHartmann_Label | anger | 1 | 0.0100 |
| synthetic_v2_review_level | Savani_Label | joy | 51 | 0.5100 |
| synthetic_v2_review_level | Savani_Label | love | 16 | 0.1600 |
| synthetic_v2_review_level | Savani_Label | fear | 13 | 0.1300 |
| synthetic_v2_review_level | Savani_Label | sadness | 10 | 0.1000 |
| synthetic_v2_review_level | Savani_Label | anger | 5 | 0.0500 |
| synthetic_v2_review_level | Savani_Label | surprise | 5 | 0.0500 |
| synthetic_v2_review_level | GoEmotions_Label | neutral | 39 | 0.3900 |
| synthetic_v2_review_level | GoEmotions_Label | realization | 18 | 0.1800 |
| synthetic_v2_review_level | GoEmotions_Label | admiration | 14 | 0.1400 |
| synthetic_v2_review_level | GoEmotions_Label | approval | 10 | 0.1000 |
| synthetic_v2_review_level | GoEmotions_Label | surprise | 6 | 0.0600 |
| synthetic_v2_review_level | GoEmotions_Label | love | 4 | 0.0400 |
| synthetic_v2_review_level | GoEmotions_Label | joy | 3 | 0.0300 |
| synthetic_v2_review_level | GoEmotions_Label | disappointment | 2 | 0.0200 |
| synthetic_v2_review_level | GoEmotions_Label | sadness | 2 | 0.0200 |
| synthetic_v2_review_level | GoEmotions_Label | desire | 1 | 0.0100 |
| synthetic_v2_review_level | GoEmotions_Label | relief | 1 | 0.0100 |
| synthetic_v2_review_level | Cardiff_Label | optimism | 93 | 0.9300 |
| synthetic_v2_review_level | Cardiff_Label | sadness | 5 | 0.0500 |
| synthetic_v2_review_level | Cardiff_Label | anger | 2 | 0.0200 |
| og_same20_perfume_level | JHartmann_Label | neutral | 12 | 0.6000 |
| og_same20_perfume_level | JHartmann_Label | joy | 8 | 0.4000 |
| og_same20_perfume_level | Savani_Label | joy | 19 | 0.9500 |
| og_same20_perfume_level | Savani_Label | love | 1 | 0.0500 |
| og_same20_perfume_level | GoEmotions_Label | neutral | 10 | 0.5000 |
| og_same20_perfume_level | GoEmotions_Label | admiration | 7 | 0.3500 |
| og_same20_perfume_level | GoEmotions_Label | approval | 1 | 0.0500 |
| og_same20_perfume_level | GoEmotions_Label | love | 1 | 0.0500 |
| og_same20_perfume_level | GoEmotions_Label | excitement | 1 | 0.0500 |
| og_same20_perfume_level | Cardiff_Label | optimism | 20 | 1.0000 |

## Interpretation

This comparison tells us whether the synthetic reviews activate the same feature-producing pipeline as OG reviews. It is not a model-accuracy comparison yet. To compare F1/ROC/PR on synthetic reviews, the pilot needs review-level labels first.
