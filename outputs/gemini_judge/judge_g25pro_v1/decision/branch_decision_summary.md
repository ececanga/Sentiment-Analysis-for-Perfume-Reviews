# Phase 1 Judge Branch Decision: go

## Decision Inputs
- Merge integrity pass: `True`
- Baseline gold macro-F1: `0.268197`
- Judge gold macro-F1: `0.432900`
- Baseline benchmark mean macro-F1: `0.238986`
- Judge benchmark mean macro-F1: `0.283992`
- Official collapsed labels: `none`

## Decision Reasons
- judge_gold_macro_f1_improved
- judge_benchmark_mean_macro_f1_improved

## Integrity Notes
- Baseline merge integrity: `{"duplicate_total": 0, "expected_rows": 784, "gemini_only": 0, "hash_mismatch_count": 0, "matched": 784, "passed": true, "unified_only": 0}`
- Judge merge integrity: `{"duplicate_total": 0, "expected_rows": 784, "gemini_only": 0, "hash_mismatch_count": 0, "matched": 784, "passed": true, "unified_only": 0}`
- Benchmark integrity: `{"baseline_complete_rows": 5, "baseline_integrity_pass": true, "baseline_mean_macro_f1": 0.238986, "baseline_thresholds_complete": [0.3], "expected_models": ["logistic_regression", "xgboost", "random_forest", "bert", "bilstm_attention"], "judge_complete_rows": 5, "judge_integrity_pass": true, "judge_mean_macro_f1": 0.28399199999999997, "judge_thresholds_complete": [0.3]}`
