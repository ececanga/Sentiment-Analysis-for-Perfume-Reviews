<!--
Date Created: 2026-06-03
-->

# Gemini 2.5 Pro Frozen-784 Results Summary

## Scope

These results summarize the frozen 784-perfume YouTube transcript dataset using the Gemini 2.5 Pro label branch. Phase2A collection remains paused as a separate, resumable supplement and is not merged into these outputs.

Gemini 2.5 Pro labels should be described as improved reference labels for evaluation and policy consistency, not as perfect ground truth.

## Dataset Integrity

The strict 784-row join succeeded.

| Check | Result |
|---|---:|
| Unified feature rows | 784 |
| Gemini label rows | 784 |
| Matched rows | 784 |
| Unified-only rows | 0 |
| Gemini-only rows | 0 |
| Duplicate unified IDs | 0 |
| Duplicate Gemini IDs | 0 |
| Hash mismatches | 0 |
| Stage8 merged/OOF matched rows | 784 |
| Stage8 target truth mismatches | 0 |

Phase2A was excluded from the main analysis. The separate Phase2A file contains 35 unique perfumes, and none of those names overlap with the 784-row branch merged table.

Primary files:

- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/gemini_unified_merged.csv`
- `outputs/gemini_judge/judge_g25pro_v1/training/stage7_gemini784_metrics.json`
- `outputs/gemini_judge/judge_g25pro_v1/analysis_out/stage8_importance_summary_report.json`

## Gemini Label Distribution

| Emotion | Positive count | Positive rate |
|---|---:|---:|
| joy | 503 | 64.16% |
| anticipation | 308 | 39.29% |
| trust | 236 | 30.10% |
| sadness | 12 | 1.53% |
| surprise | 10 | 1.28% |
| disgust | 8 | 1.02% |
| fear | 3 | 0.38% |
| anger | 1 | 0.13% |

The label set is highly imbalanced. Joy, anticipation, and trust dominate the label space. Anger is not estimable, and fear/disgust should be treated as exploratory rare-label findings. This imbalance is important when interpreting macro-F1 and per-emotion metrics.

Gemini VAD target means:

| Dimension | Mean | Std. dev. |
|---|---:|---:|
| valence | 6.679 | 1.091 |
| arousal | 5.950 | 0.749 |
| dominance | 5.916 | 0.710 |

## Agreement Analysis

### NRC vs Gemini

Binary NRC-vs-Gemini agreement is low overall, with mean binary accuracy of 0.184. This is largely because the simple NRC rule marks an emotion as present whenever any matching lexicon word appears, producing many positives for nearly every emotion.

| Emotion | Accuracy | Notes |
|---|---:|---|
| joy | 0.640 | Highest binary agreement, consistent with joy being frequent. |
| anticipation | 0.393 | Moderate agreement relative to other emotions. |
| trust | 0.305 | Some alignment, but many NRC positives. |
| sadness | 0.036 | Very low, dominated by NRC positive / Gemini negative cases. |
| fear | 0.028 | Very low and rare in Gemini labels. |
| surprise | 0.024 | Very low and rare in Gemini labels. |
| anger | 0.023 | Not estimable as a reliable target. |
| disgust | 0.026 | Rare-label result; interpret cautiously. |

Interpretation: NRC captures broad lexical emotional vocabulary, but it does not map cleanly onto perfume-level Gemini labels without normalization, weighting, or calibration.

### Top-Label Agreement

Top-label agreement with Gemini primary labels was strongest for Cardiff and Savani, followed by GoEmotions and JHartmann.

| Model label source | Compared rows | Matches | Match rate |
|---|---:|---:|---:|
| Cardiff | 772 | 490 | 0.635 |
| Savani | 772 | 487 | 0.631 |
| GoEmotions | 772 | 454 | 0.588 |
| JHartmann | 772 | 391 | 0.506 |

Interpretation: transformer-based top labels are closer to Gemini than raw NRC binary labels, but agreement is still partial. The systems are measuring related but non-identical emotional signals.

### VAD Agreement

Continuous VAD agreement between lexicon-derived VAD and Gemini VAD is positive but modest.

| Dimension | Pearson r | Spearman r | MAE |
|---|---:|---:|---:|
| valence | 0.252 | 0.221 | 6.512 |
| arousal | 0.218 | 0.214 | 6.049 |
| dominance | 0.268 | 0.235 | 5.891 |

Interpretation: rank and linear correlations suggest weak-to-moderate directional alignment. The large MAE should not be overinterpreted because the compared VAD sources are on different practical scales.

## Benchmark Model Results

Official benchmark metrics use the seven official emotion targets: joy, sadness, fear, trust, surprise, anticipation, and disgust. Anger is excluded from the official benchmark because it has only one positive case in this branch.

| Model | Macro-F1 | Macro ROC-AUC | Macro PR-AUC |
|---|---:|---:|---:|
| XGBoost | 0.318 | 0.741 | 0.303 |
| Logistic Regression | 0.318 | 0.729 | 0.326 |
| BiLSTM + Attention | 0.270 | 0.488 | 0.257 |
| BERT | 0.258 | 0.543 | 0.219 |
| Random Forest | 0.256 | 0.729 | 0.297 |

Best model by macro-F1: XGBoost, narrowly ahead of Logistic Regression.

Best model by macro PR-AUC: Logistic Regression.

Strongest emotions across benchmark models:

- Joy: highest and most stable F1 across models, mean F1 0.782.
- Anticipation: second strongest, mean F1 0.570.
- Trust: moderate, mean F1 0.436.

Weakest emotions:

- Fear: near-zero F1 across benchmark models due to only 3 positives.
- Disgust, surprise, and sadness: low F1 because of severe rarity, despite some useful ROC-AUC/PR-AUC signals in certain models.

## Stage7 Branch-Local Training

The branch-local Stage7 training used 784 rows and 58 numeric features from NRC, VAD, Hartmann, Savani, GoEmotions, and Cardiff feature families.

Global emotion metrics:

| Metric | Value |
|---|---:|
| Macro-F1 over nondegenerate labels | 0.301 |
| Micro-F1 over nondegenerate labels | 0.514 |
| Subset accuracy | 0.200 |
| Hamming loss | 0.180 |
| Label cardinality | 1.379 |

Per-label Stage7 diagnostics:

| Emotion | Positives | Status | F1 | ROC-AUC | Average precision |
|---|---:|---|---:|---:|---:|
| joy | 503 | trained | 0.675 | 0.651 | 0.734 |
| anticipation | 308 | trained | 0.492 | 0.608 | 0.511 |
| trust | 236 | trained | 0.360 | 0.511 | 0.308 |
| fear | 3 | trained, rare | 0.286 | 0.995 | 0.528 |
| disgust | 8 | trained, rare | 0.222 | 0.895 | 0.224 |
| sadness | 12 | trained | 0.218 | 0.868 | 0.153 |
| surprise | 10 | trained | 0.154 | 0.754 | 0.097 |
| anger | 1 | insufficient minority | 0.000 | 0.500 | 0.001 |

The rare-label ROC-AUC values can look high because there are very few positive examples. F1 and support counts are more important for practical interpretation.

Stage7 VAD prediction:

| Dimension | MAE | RMSE | R2 | Pearson r | Spearman r |
|---|---:|---:|---:|---:|---:|
| valence | 0.787 | 1.021 | 0.123 | 0.407 | 0.387 |
| arousal | 0.569 | 0.724 | 0.065 | 0.335 | 0.301 |
| dominance | 0.550 | 0.733 | -0.066 | 0.226 | 0.228 |
| average | 0.635 | 0.826 | 0.041 | 0.323 | 0.305 |

VAD prediction is more stable than the rare discrete emotions but still modest, especially for dominance.

## Stage8 Feature Importance

Stage8 used strict branch-local alignment: 784 merged rows, 784 OOF rows, zero ID mismatches, and zero target truth mismatches.

Headline emotion feature-family importance over non-rare headline targets:

| Family | Mean positive importance |
|---|---:|
| Hartmann | 0.0779 |
| GoEmotions | 0.0443 |
| Cardiff | 0.0366 |
| Savani | 0.0137 |
| VAD | 0.0028 |
| NRC | 0.0000 |

Top individual emotion features:

| Feature | Mean positive importance |
|---|---:|
| GoEmotions_excitement | 0.0484 |
| Hartmann_sadness | 0.0437 |
| Cardiff_sadness | 0.0286 |
| Hartmann_surprise | 0.0269 |
| Savani_anger | 0.0153 |
| NRC_negative | 0.0113 |
| VAD_valence | 0.0112 |
| GoEmotions_love | 0.0111 |
| Savani_fear | 0.0110 |
| NRC_anticipation | 0.0104 |

For emotion prediction, transformer-derived feature families carry most of the predictive signal. NRC and VAD features add some localized signal but are weaker as headline predictors for discrete Gemini labels.

Headline VAD family importance:

| Family | Mean positive importance |
|---|---:|
| GoEmotions | 0.0686 |
| Cardiff | 0.0248 |
| VAD | 0.0144 |
| NRC | 0.0075 |
| Hartmann | 0.0045 |
| Savani | 0.0018 |

Top individual VAD features:

| Feature | Mean positive importance |
|---|---:|
| NRC_joy | 0.1226 |
| NRC_positive | 0.0798 |
| NRC_fear | 0.0727 |
| NRC_trust | 0.0697 |
| NRC_anticipation | 0.0452 |
| NRC_negative | 0.0379 |
| NRC_anger | 0.0365 |
| NRC_surprise | 0.0353 |
| GoEmotions_excitement | 0.0214 |
| NRC_disgust | 0.0176 |

For VAD prediction, individual NRC affect-count features are prominent, while GoEmotions is the strongest grouped family. This suggests that broad lexical affect counts are useful for continuous affect dimensions even when they are less reliable for exact discrete emotion labels.

## Thesis-Ready Findings

- The frozen 784 dataset joined cleanly with the Gemini 2.5 Pro label branch: all 784 rows matched, with no duplicate IDs, no unmatched rows, and no hash mismatches.
- Phase2A was not merged into the main experiments. It remains a separate 35-perfume supplement because quota/IP constraints made the 2K expansion infeasible during the thesis timeline.
- Gemini 2.5 Pro labels are highly imbalanced: joy, anticipation, and trust dominate, while anger, fear, disgust, surprise, and sadness are rare.
- The strongest predictive performance is on common emotions, especially joy and anticipation. Rare labels should be discussed cautiously because there are too few positives for stable estimation.
- XGBoost gives the best macro-F1, but Logistic Regression is essentially tied and has the best macro PR-AUC. This supports a conservative interpretation: classical tabular models remain competitive on this feature set.
- Neural text models underperform the best classical tabular models in this frozen setup, likely because the dataset is small and label imbalance is severe.
- Transformer-derived feature families, especially Hartmann and GoEmotions, are the strongest contributors for discrete emotion prediction.
- NRC features are weak for direct binary emotion agreement with Gemini labels, but individual NRC affect counts are useful for VAD prediction.
- Lexicon VAD and Gemini VAD show positive but modest correlations, suggesting they capture related affective directions but are not interchangeable scales.
- Gemini 2.5 Pro labels should be framed as a stronger reference-label branch for evaluation, not as perfect human ground truth.

## Limitations

- Gemini labels are model-generated reference labels, not human consensus labels.
- The label distribution is extremely imbalanced; anger, fear, disgust, surprise, and sadness have limited positive support.
- Macro metrics are sensitive to rare labels, and rare-label ROC-AUC can be misleading when support is very small.
- NRC binary agreement uses a simple `NRC_count > 0` rule and should not be treated as a calibrated classifier.
- VAD agreement MAE between lexicon and Gemini outputs is affected by scale differences.
- The benchmark reflects the frozen 784 transcript corpus, not the intended 2K expansion.
- Phase2A is useful as a future collection branch or qualitative supplement, but it is not part of the quantitative main results.

## Suggested Phase2A Wording

The planned 2,000-perfume YouTube transcript expansion was paused because YouTube API quota limits, transcript retrieval IP blocking, low safe-collection yield, and thesis timeline constraints made the full expansion infeasible. The Phase2A artifacts were preserved as a resumable supplement, while all main scoring and evaluation results use the frozen 784-perfume transcript dataset with Gemini 2.5 Pro reference labels.

## Recommended Next Writing Step

Write the Results section around three tables:

1. Dataset and label distribution table.
2. Benchmark model comparison table.
3. Stage8 feature-family importance table.

Then use the Discussion section to explain why common affective labels are more stable, why rare labels limit macro performance, and why Phase2A remains separate.
