# Historical seed-42 reuse audit

All entries were checked against the fixed dataset, labels, source groups, outer/inner splits, model settings, preprocessing, threshold, and complete predictions. Deep loaders also validate text and checkpoint provenance. Historical files remain unchanged.

| Job | Decision | Reason |
| --- | --- | --- |
| main/logistic_regression/youtube_only/full_hybrid/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| main/logistic_regression/youtube_plus_weighted_synthetic_784x3/full_hybrid/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| main/xgboost/youtube_only/full_hybrid/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| main/xgboost/youtube_plus_weighted_synthetic_784x3/full_hybrid/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| main/random_forest/youtube_only/full_hybrid/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| main/random_forest/youtube_plus_weighted_synthetic_784x3/full_hybrid/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| main/bert/youtube_only/standard/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| main/bert/youtube_plus_weighted_synthetic_784x3/standard/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| main/bilstm_attention/youtube_only/standard/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| main/bilstm_attention/youtube_plus_weighted_synthetic_784x3/standard/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/full_hybrid/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/nrc_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/vad_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/lexicon_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/hartmann_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/savani_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/goemotions_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/cardiff_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/transformer_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/no_nrc/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/no_vad/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/no_hartmann/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/no_savani/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/no_goemotions/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/logistic_regression/youtube_only/no_cardiff/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/xgboost/youtube_only/full_hybrid/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/xgboost/youtube_only/nrc_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/xgboost/youtube_only/vad_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/xgboost/youtube_only/transformer_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/xgboost/youtube_only/nrc_plus_vad/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/xgboost/youtube_only/nrc_plus_transformer/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/xgboost/youtube_only/vad_plus_transformer/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/random_forest/youtube_only/full_hybrid/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/random_forest/youtube_only/nrc_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/random_forest/youtube_only/vad_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/random_forest/youtube_only/transformer_only/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/random_forest/youtube_only/nrc_plus_vad/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/random_forest/youtube_only/nrc_plus_transformer/data42_model42 | reusable | Frozen protocol and complete predictions validated |
| structured_ablation/random_forest/youtube_only/vad_plus_transformer/data42_model42 | reusable | Frozen protocol and complete predictions validated |

Historical import applies only to seed 42. Other seeds require training only when a matching validated completed run is absent. Reuse is opt-in with `--reuse-seed42`; the JSON audit retains source paths and hashes.
