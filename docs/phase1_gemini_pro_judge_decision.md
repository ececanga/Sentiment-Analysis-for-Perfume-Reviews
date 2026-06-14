# Phase 1 Gemini 2.5 Pro Judge Decision

Freeze generated (UTC): `2026-05-24T11:57:26.007167+00:00`

## Project Context

This repo has a frozen 784-perfume baseline built with Gemini 2.5 Flash labels and downstream benchmark artifacts. The canonical frozen baseline label artifact remains `labels/gemini_labels_784_v1.csv` and must stay immutable.

## Why Phase 1 Existed

Phase 1 was a controlled judge-branch comparison to decide whether the future expansion path should keep the frozen Gemini 2.5 Flash labeling policy or switch to Gemini 2.5 Pro. The branch design held the dataset, raw review and transcript universe, snippet selection policy, strict prompt and schema, aggregation rule, folds, seeds, and benchmark protocol fixed. The only intended variable was the judge model.

## Branch Definitions

- Frozen baseline branch: same 784 perfumes, frozen labels from Gemini 2.5 Flash, canonical label CSV `labels/gemini_labels_784_v1.csv`.
- Gemini Pro judge branch: same 784 perfumes and same evaluation contract, but the labeling judge changed to Gemini 2.5 Pro under branch ID `judge_g25pro_v1`.

## Phase 1 Branch Files And Artifacts

Implemented branch files:

- `config/gemini_judge/judge_g25pro_v1.yaml`
- `src/pipeline/run_gemini_judge_branch.py`
- `src/analysis/compare_label_branch_to_gold43.py`
- `src/analysis/compare_gemini_branches.py`
- `scripts/run_phase1_judge.sh`

Frozen branch artifacts:

- Output root: `outputs/gemini_judge/judge_g25pro_v1`
- Decision summary JSON: `outputs/gemini_judge/judge_g25pro_v1/decision/branch_decision_summary.json`
- Freeze manifest JSON: `outputs/gemini_judge/judge_g25pro_v1/phase1_freeze_manifest.json`
- Baseline label CSV: `labels/gemini_labels_784_v1.csv`
- Judge label CSV: `labels/gemini_labels_784_judge_g25pro_v1.csv`
- Judge label JSONL: `labels/gemini_labels_784_judge_g25pro_v1.jsonl`
- Judge labeling report JSON: `labels/gemini_labeling_report_judge_g25pro_v1.json`
- Judge labeling errors JSONL: `labels/gemini_labeling_errors_judge_g25pro_v1.jsonl`
- Judge labeling run log: `labels/gemini_labeling_run_judge_g25pro_v1.log`
- Baseline gold metrics JSON: `outputs/gemini_judge/judge_g25pro_v1/gold_eval/baseline_vs_gold_metrics.json`
- Judge gold metrics JSON: `outputs/gemini_judge/judge_g25pro_v1/gold_eval/judge_vs_gold_metrics.json`
- Judge benchmark comparison table: `outputs/gemini_judge/judge_g25pro_v1/model_benchmark/tables/comparison_table.csv`

## Final Decision

`GO`

Gemini 2.5 Pro won Phase 1 under the repo decision rule and should be promoted as the labeling policy candidate for the next expansion stage.

## Key Metrics

| Metric | Frozen baseline: Gemini 2.5 Flash | Judge branch: Gemini 2.5 Pro |
| --- | ---: | ---: |
| Gold macro-F1 | 0.26819727891156464 | 0.4329004329004329 |
| Benchmark mean macro-F1 | 0.238986 | 0.28399199999999997 |

## Integrity Checks Summary

- Final decision in decision summary: `go`
- `merge_integrity_pass`: `true`
- `official_collapsed_labels`: `[]`
- Baseline benchmark complete rows: `5`
- Judge benchmark complete rows: `5`
- Baseline threshold: `[0.3]`
- Judge threshold: `[0.3]`
- Baseline label CSV rows: `784`
- Judge label CSV rows: `784`
- Judge label JSONL rows: `784`
- Judge JSONL metadata model: `gemini-2.5-pro`
- Baseline and judge `perfume_id` sets match: `true`
- Judge CSV and JSONL `perfume_id` sets match: `true`
- Baseline merge duplicates: `0`
- Baseline merge hash mismatches: `0`
- Judge merge duplicates: `0`
- Judge merge hash mismatches: `0`
- Judge labeling failures: `0`
- Judge missing IDs: `0`
- Key artifact SHA-256 hashes captured in manifest: `16`
- Branch outputs remain isolated under `outputs/gemini_judge/judge_g25pro_v1`

## Known Limitations

- Phase 1 answers a branch policy question, not whether the labels are perfect ground truth.
- The matched gold evaluation set remains small at `43` perfumes.
- Some official labels remain weak on gold evaluation, especially sadness and surprise. Judge per-label F1 on the gold set was sadness `0` and surprise `0`.
- Anger was excluded from the primary decision metric and should not be over-interpreted from this Phase 1 decision alone.

## Conclusion

Promote the Gemini 2.5 Pro labeling policy to the next controlled expansion stage. The Phase 1 result is a documented `GO`, with better gold macro-F1 and better benchmark mean macro-F1 than the frozen Gemini 2.5 Flash baseline, while preserving the branch integrity checks required for the 784-perfume comparison.

## Next Phase

Build a versioned 2,000-perfume expansion path using the Gemini 2.5 Pro policy before attempting the later 5,000-perfume scale-up. Do not reuse `judge_g25pro_v1` output roots. Use a new config and output root, likely:

- `config/gemini_judge/judge_g25pro_2k_v1.yaml`
- `outputs/gemini_judge/judge_g25pro_2k_v1/`

Warnings: None.
