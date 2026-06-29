# Synthetic-Only 784x3 Data-Level Resampling Ablation

This compares imbalanced-learn random oversampling, random undersampling, and SMOTE against unbalanced and model-weighted baselines.

- Rows: 2352
- Unique perfumes: 784
- Reviews per perfume: 3
- Review indices: 1, 2, 3
- Threshold: 0.3
- imbalanced-learn version: 0.14.2
- Split: GroupKFold with 5 folds, grouped by perfume_id
- Resampling is done inside each training fold separately, for each binary label classifier.
- Test folds are never resampled.

## Metric Summary

| model_family | strategy | resampling_method | runs | macro_f1_mean | macro_f1_std | delta_macro_f1_mean_vs_unbalanced | macro_roc_auc_mean | macro_pr_auc_mean | predicted_label_cardinality_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | unbalanced | none | 5 | 0.6514 | 0.0000 | 0.0000 | 0.9428 | 0.6598 | 2.0425 |
| Logistic Regression | smote | smote | 5 | 0.6358 | 0.0013 | -0.0155 | 0.9331 | 0.6443 | 2.4207 |
| Logistic Regression | oversampling | oversampling | 5 | 0.6319 | 0.0013 | -0.0195 | 0.9419 | 0.6459 | 2.5082 |
| Logistic Regression | class_weight | none | 5 | 0.6277 | 0.0000 | -0.0237 | 0.9424 | 0.6497 | 2.5395 |
| Logistic Regression | undersampling | undersampling | 5 | 0.5946 | 0.0055 | -0.0567 | 0.9312 | 0.6073 | 2.7014 |
| XGBoost | scale_pos_weight | none | 5 | 0.6624 | 0.0046 | 0.0048 | 0.9483 | 0.6745 | 2.1228 |
| XGBoost | oversampling | oversampling | 5 | 0.6581 | 0.0064 | 0.0004 | 0.9482 | 0.6719 | 2.1188 |
| XGBoost | smote | smote | 5 | 0.6577 | 0.0020 | 0.0001 | 0.9470 | 0.6766 | 2.1742 |
| XGBoost | unbalanced | none | 5 | 0.6576 | 0.0056 | 0.0000 | 0.9498 | 0.6822 | 1.9492 |
| XGBoost | undersampling | undersampling | 5 | 0.6168 | 0.0033 | -0.0408 | 0.9449 | 0.6649 | 2.5472 |

## Per-Label Metrics

| model_family | strategy | label | positive_rate | f1_mean | delta_f1_mean_vs_unbalanced | pr_auc_mean | delta_pr_auc_mean_vs_unbalanced | predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | class_weight | anticipation | 0.7798 | 0.9515 | 0.0023 | 0.9876 | -0.0000 | 0.7878 |
| Logistic Regression | oversampling | anticipation | 0.7798 | 0.9509 | 0.0018 | 0.9875 | -0.0001 | 0.7856 |
| Logistic Regression | smote | anticipation | 0.7798 | 0.9471 | -0.0021 | 0.9869 | -0.0007 | 0.7873 |
| Logistic Regression | unbalanced | anticipation | 0.7798 | 0.9492 | 0.0000 | 0.9876 | 0.0000 | 0.8265 |
| Logistic Regression | undersampling | anticipation | 0.7798 | 0.9459 | -0.0033 | 0.9863 | -0.0013 | 0.7741 |
| Logistic Regression | class_weight | disgust | 0.0098 | 0.2317 | 0.0779 | 0.1314 | -0.0305 | 0.0599 |
| Logistic Regression | oversampling | disgust | 0.0098 | 0.2500 | 0.0961 | 0.1319 | -0.0299 | 0.0508 |
| Logistic Regression | smote | disgust | 0.0098 | 0.1994 | 0.0456 | 0.1123 | -0.0496 | 0.0499 |
| Logistic Regression | unbalanced | disgust | 0.0098 | 0.1538 | 0.0000 | 0.1619 | 0.0000 | 0.0068 |
| Logistic Regression | undersampling | disgust | 0.0098 | 0.1029 | -0.0509 | 0.0665 | -0.0954 | 0.1771 |
| Logistic Regression | class_weight | fear | 0.0940 | 0.6386 | -0.0800 | 0.6367 | 0.0077 | 0.1884 |
| Logistic Regression | oversampling | fear | 0.0940 | 0.6476 | -0.0709 | 0.6260 | -0.0030 | 0.1826 |
| Logistic Regression | smote | fear | 0.0940 | 0.6852 | -0.0333 | 0.6276 | -0.0014 | 0.1667 |
| Logistic Regression | unbalanced | fear | 0.0940 | 0.7185 | 0.0000 | 0.6290 | 0.0000 | 0.1356 |
| Logistic Regression | undersampling | fear | 0.0940 | 0.6006 | -0.1179 | 0.5522 | -0.0768 | 0.2068 |
| Logistic Regression | class_weight | joy | 0.1327 | 0.4694 | -0.1226 | 0.6086 | -0.0240 | 0.3746 |
| Logistic Regression | oversampling | joy | 0.1327 | 0.4678 | -0.1242 | 0.6031 | -0.0295 | 0.3716 |
| Logistic Regression | smote | joy | 0.1327 | 0.5005 | -0.0915 | 0.5888 | -0.0437 | 0.3318 |
| Logistic Regression | unbalanced | joy | 0.1327 | 0.5920 | 0.0000 | 0.6326 | 0.0000 | 0.1331 |
| Logistic Regression | undersampling | joy | 0.1327 | 0.4502 | -0.1418 | 0.5388 | -0.0938 | 0.3936 |
| Logistic Regression | class_weight | sadness | 0.0676 | 0.4795 | -0.0231 | 0.3996 | -0.0203 | 0.2020 |
| Logistic Regression | oversampling | sadness | 0.0676 | 0.4813 | -0.0213 | 0.3912 | -0.0287 | 0.1963 |
| Logistic Regression | smote | sadness | 0.0676 | 0.4889 | -0.0137 | 0.4126 | -0.0073 | 0.1818 |
| Logistic Regression | unbalanced | sadness | 0.0676 | 0.5026 | 0.0000 | 0.4199 | 0.0000 | 0.0948 |
| Logistic Regression | undersampling | sadness | 0.0676 | 0.4408 | -0.0618 | 0.3317 | -0.0882 | 0.2261 |
| Logistic Regression | class_weight | surprise | 0.3291 | 0.8192 | 0.0057 | 0.8791 | -0.0024 | 0.4588 |
| Logistic Regression | oversampling | surprise | 0.3291 | 0.8203 | 0.0069 | 0.8783 | -0.0031 | 0.4566 |
| Logistic Regression | smote | surprise | 0.3291 | 0.8172 | 0.0037 | 0.8772 | -0.0043 | 0.4511 |
| Logistic Regression | unbalanced | surprise | 0.3291 | 0.8135 | 0.0000 | 0.8815 | 0.0000 | 0.4277 |
| Logistic Regression | undersampling | surprise | 0.3291 | 0.8161 | 0.0026 | 0.8725 | -0.0090 | 0.4576 |
| Logistic Regression | class_weight | trust | 0.3546 | 0.8041 | -0.0258 | 0.9049 | -0.0015 | 0.4681 |
| Logistic Regression | oversampling | trust | 0.3546 | 0.8053 | -0.0247 | 0.9031 | -0.0033 | 0.4646 |
| Logistic Regression | smote | trust | 0.3546 | 0.8124 | -0.0176 | 0.9051 | -0.0013 | 0.4520 |
| Logistic Regression | unbalanced | trust | 0.3546 | 0.8299 | 0.0000 | 0.9064 | 0.0000 | 0.4179 |
| Logistic Regression | undersampling | trust | 0.3546 | 0.8058 | -0.0241 | 0.9029 | -0.0036 | 0.4660 |
| XGBoost | oversampling | anticipation | 0.7798 | 0.9596 | 0.0018 | 0.9913 | -0.0003 | 0.8007 |
| XGBoost | scale_pos_weight | anticipation | 0.7798 | 0.9591 | 0.0013 | 0.9915 | -0.0000 | 0.8002 |
| XGBoost | smote | anticipation | 0.7798 | 0.9583 | 0.0005 | 0.9915 | -0.0001 | 0.7977 |
| XGBoost | unbalanced | anticipation | 0.7798 | 0.9578 | 0.0000 | 0.9916 | 0.0000 | 0.8149 |
| XGBoost | undersampling | anticipation | 0.7798 | 0.9463 | -0.0115 | 0.9909 | -0.0007 | 0.7595 |
| XGBoost | oversampling | disgust | 0.0098 | 0.1800 | 0.0186 | 0.1428 | -0.0612 | 0.0158 |
| XGBoost | scale_pos_weight | disgust | 0.0098 | 0.2064 | 0.0449 | 0.1756 | -0.0283 | 0.0141 |
| XGBoost | smote | disgust | 0.0098 | 0.2214 | 0.0599 | 0.1993 | -0.0047 | 0.0232 |
| XGBoost | unbalanced | disgust | 0.0098 | 0.1615 | 0.0000 | 0.2040 | 0.0000 | 0.0039 |
| XGBoost | undersampling | disgust | 0.0098 | 0.1173 | -0.0442 | 0.1946 | -0.0094 | 0.1526 |
| XGBoost | oversampling | fear | 0.0940 | 0.7059 | -0.0108 | 0.6417 | -0.0076 | 0.1359 |
| XGBoost | scale_pos_weight | fear | 0.0940 | 0.7154 | -0.0012 | 0.6361 | -0.0132 | 0.1376 |
| XGBoost | smote | fear | 0.0940 | 0.7007 | -0.0159 | 0.6558 | 0.0065 | 0.1424 |
| XGBoost | unbalanced | fear | 0.0940 | 0.7166 | 0.0000 | 0.6493 | 0.0000 | 0.1194 |
| XGBoost | undersampling | fear | 0.0940 | 0.6372 | -0.0795 | 0.6072 | -0.0421 | 0.1883 |
| XGBoost | oversampling | joy | 0.1327 | 0.6007 | -0.0204 | 0.6777 | -0.0015 | 0.1830 |
| XGBoost | scale_pos_weight | joy | 0.1327 | 0.5886 | -0.0326 | 0.6787 | -0.0004 | 0.1849 |
| XGBoost | smote | joy | 0.1327 | 0.5596 | -0.0615 | 0.6651 | -0.0140 | 0.2144 |
| XGBoost | unbalanced | joy | 0.1327 | 0.6211 | 0.0000 | 0.6791 | 0.0000 | 0.1088 |
| XGBoost | undersampling | joy | 0.1327 | 0.4874 | -0.1337 | 0.6441 | -0.0350 | 0.3437 |
| XGBoost | oversampling | sadness | 0.0676 | 0.5071 | 0.0146 | 0.4409 | -0.0029 | 0.1249 |
| XGBoost | scale_pos_weight | sadness | 0.0676 | 0.5121 | 0.0196 | 0.4319 | -0.0119 | 0.1227 |
| XGBoost | smote | sadness | 0.0676 | 0.5107 | 0.0182 | 0.4177 | -0.0260 | 0.1349 |
| XGBoost | unbalanced | sadness | 0.0676 | 0.4925 | 0.0000 | 0.4438 | 0.0000 | 0.0770 |
| XGBoost | undersampling | sadness | 0.0676 | 0.4864 | -0.0061 | 0.4145 | -0.0293 | 0.2016 |
| XGBoost | oversampling | surprise | 0.3291 | 0.8240 | 0.0012 | 0.8841 | 0.0011 | 0.4403 |
| XGBoost | scale_pos_weight | surprise | 0.3291 | 0.8238 | 0.0010 | 0.8819 | -0.0011 | 0.4428 |
| XGBoost | smote | surprise | 0.3291 | 0.8243 | 0.0014 | 0.8846 | 0.0016 | 0.4422 |
| XGBoost | unbalanced | surprise | 0.3291 | 0.8228 | 0.0000 | 0.8830 | 0.0000 | 0.4245 |
| XGBoost | undersampling | surprise | 0.3291 | 0.8209 | -0.0019 | 0.8806 | -0.0024 | 0.4557 |
| XGBoost | oversampling | trust | 0.3546 | 0.8290 | -0.0019 | 0.9246 | -0.0003 | 0.4182 |
| XGBoost | scale_pos_weight | trust | 0.3546 | 0.8314 | 0.0005 | 0.9258 | 0.0009 | 0.4205 |
| XGBoost | smote | trust | 0.3546 | 0.8290 | -0.0019 | 0.9220 | -0.0028 | 0.4195 |
| XGBoost | unbalanced | trust | 0.3546 | 0.8309 | 0.0000 | 0.9249 | 0.0000 | 0.4007 |
| XGBoost | undersampling | trust | 0.3546 | 0.8225 | -0.0084 | 0.9225 | -0.0023 | 0.4458 |

## Label Distribution

| label | positive_count | positive_rate | neg_to_pos_ratio |
| --- | --- | --- | --- |
| joy | 312 | 0.1327 | 6.5385 |
| sadness | 159 | 0.0676 | 13.7925 |
| fear | 221 | 0.0940 | 9.6425 |
| trust | 834 | 0.3546 | 1.8201 |
| surprise | 774 | 0.3291 | 2.0388 |
| anticipation | 1834 | 0.7798 | 0.2824 |
| disgust | 23 | 0.0098 | 101.2609 |
