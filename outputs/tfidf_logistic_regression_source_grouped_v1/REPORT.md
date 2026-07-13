# TF-IDF + Logistic Regression Source-Grouped Baseline

This simple text baseline uses transcript-only perfume documents, TF-IDF unigrams and bigrams fit inside each outer training fold, and unweighted binary Logistic Regression. It uses the same 784 perfumes and repaired five-fold source-component manifest as the accepted benchmark.

| Metric | Value |
| --- | ---: |
| Macro-F1 at threshold 0.3 | 0.3254 |
| Macro ROC-AUC | 0.5409 |
| Macro average precision | 0.3095 |
| Micro-F1 | 0.7525 |
| Hamming loss | 0.1755 |
| Mean predicted labels per perfume | 2.9974 |

This row is a simple learned text baseline. It is not a sixth primary model and was added after the repaired five-model benchmark was inspected.
