"""Freeze and document the completed Phase 1 Gemini 2.5 Pro judge branch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.paths import PROJECT_ROOT


EXPECTED_BRANCH_ID = "judge_g25pro_v1"
EXPECTED_JUDGE_MODEL = "gemini-2.5-pro"
EXPECTED_ROW_COUNT = 784
EXPECTED_GOLD_ROWS = 43
EXPECTED_THRESHOLD = 0.3
EXPECTED_BENCHMARK_MODELS = (
    "logistic_regression",
    "xgboost",
    "random_forest",
    "bert",
    "bilstm_attention",
)
EXPECTED_PRIMARY_EXCLUDED_LABELS = ("anger",)

IMPLEMENTATION_FILES = (
    "config/gemini_judge/judge_g25pro_v1.yaml",
    "src/pipeline/run_gemini_judge_branch.py",
    "src/analysis/compare_label_branch_to_gold43.py",
    "src/analysis/compare_gemini_branches.py",
    "scripts/run_phase1_judge.sh",
)

DEFAULT_DECISION_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "gemini_judge"
    / EXPECTED_BRANCH_ID
    / "decision"
    / "branch_decision_summary.json"
)
DEFAULT_OUT_JSON = (
    PROJECT_ROOT / "outputs" / "gemini_judge" / EXPECTED_BRANCH_ID / "phase1_freeze_manifest.json"
)
DEFAULT_OUT_MD = PROJECT_ROOT / "docs" / "phase1_gemini_pro_judge_decision.md"


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the completed Phase 1 Gemini 2.5 Pro judge branch."
    )
    parser.add_argument(
        "--decision-json",
        type=Path,
        default=DEFAULT_DECISION_JSON,
        help="Completed Phase 1 branch decision summary JSON.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help="Output path for the machine-readable Phase 1 freeze manifest JSON.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_OUT_MD,
        help="Output path for the human-readable Phase 1 decision note Markdown.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expect_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be a JSON object.")
    return value


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return expect_mapping(payload, label=f"Invalid JSON payload in {path}")


def resolve_repo_path(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot interpret boolean value {value!r}.")


def as_float(value: Any, *, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, received {value!r}.") from exc


def as_int(value: Any, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, received {value!r}.") from exc


def nearly_equal(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_csv_perfume_ids(path: Path) -> tuple[int, set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"CSV has no header row: {path}")
        require("perfume_id" in reader.fieldnames, f"CSV is missing perfume_id column: {path}")

        count = 0
        seen: set[str] = set()
        duplicates: set[str] = set()
        for count, row in enumerate(reader, start=1):
            perfume_id = (row.get("perfume_id") or "").strip()
            require(perfume_id != "", f"Blank perfume_id detected in {path} at data row {count}.")
            if perfume_id in seen:
                duplicates.add(perfume_id)
            seen.add(perfume_id)

    require(not duplicates, f"Duplicate perfume_id values in {path}: {sorted(duplicates)[:10]}")
    return count, seen


def collect_jsonl_perfume_ids(path: Path, *, expected_model: str) -> tuple[int, set[str]]:
    count = 0
    seen: set[str] = set()
    duplicates: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            require(isinstance(payload, dict), f"JSONL row {line_number} in {path} is not an object.")

            perfume_id = str(payload.get("perfume_id", "")).strip()
            require(perfume_id != "", f"Blank perfume_id in JSONL row {line_number}: {path}")
            if perfume_id in seen:
                duplicates.add(perfume_id)
            seen.add(perfume_id)

            metadata = payload.get("metadata")
            require(isinstance(metadata, dict), f"Missing metadata object in JSONL row {line_number}: {path}")
            model = str(metadata.get("model", "")).strip()
            require(
                model == expected_model,
                f"Unexpected metadata.model in JSONL row {line_number}: expected {expected_model}, found {model!r}.",
            )
            count += 1

    require(not duplicates, f"Duplicate perfume_id values in {path}: {sorted(duplicates)[:10]}")
    return count, seen


def validate_labeling_report(path: Path) -> Mapping[str, Any]:
    payload = load_json(path)
    counts = expect_mapping(payload.get("counts"), label=f"counts section in {path}")

    total_targets = as_int(counts.get("total_targets"), label=f"counts.total_targets in {path}")
    succeeded = as_int(counts.get("succeeded"), label=f"counts.succeeded in {path}")
    failed = as_int(counts.get("failed"), label=f"counts.failed in {path}")
    missing_ids = payload.get("missing_ids")
    failed_ids = payload.get("failed_ids")

    require(total_targets == EXPECTED_ROW_COUNT, f"Expected {EXPECTED_ROW_COUNT} total targets in {path}.")
    require(succeeded == EXPECTED_ROW_COUNT, f"Expected {EXPECTED_ROW_COUNT} succeeded rows in {path}.")
    require(failed == 0, f"Expected zero failed rows in {path}.")
    require(missing_ids == [], f"Expected empty missing_ids in {path}, found {missing_ids!r}.")
    require(failed_ids == [], f"Expected empty failed_ids in {path}, found {failed_ids!r}.")
    return payload


def validate_gold_metrics(
    path: Path,
    *,
    expected_branch_name: str,
    expected_macro_f1: float,
) -> Mapping[str, Any]:
    payload = load_json(path)
    branch_name = str(payload.get("branch_name", "")).strip()
    require(branch_name == expected_branch_name, f"Unexpected branch_name in {path}: {branch_name!r}")

    matched_gold_rows = as_int(payload.get("matched_gold_rows"), label=f"matched_gold_rows in {path}")
    require(matched_gold_rows == EXPECTED_GOLD_ROWS, f"Expected {EXPECTED_GOLD_ROWS} matched gold rows in {path}.")

    primary = expect_mapping(payload.get("primary_decision_metric"), label=f"primary_decision_metric in {path}")
    primary_name = str(primary.get("name", "")).strip()
    label_scope = str(primary.get("label_scope", "")).strip()
    excluded_labels = tuple(primary.get("excluded_labels", []))
    value = as_float(primary.get("value"), label=f"primary_decision_metric.value in {path}")

    require(primary_name == "macro_f1", f"Unexpected primary metric name in {path}: {primary_name!r}")
    require(
        label_scope == "official_target_labels",
        f"Unexpected primary metric label scope in {path}: {label_scope!r}",
    )
    require(
        excluded_labels == EXPECTED_PRIMARY_EXCLUDED_LABELS,
        f"Unexpected excluded labels in {path}: {excluded_labels!r}",
    )
    require(
        nearly_equal(value, expected_macro_f1),
        f"Gold macro-F1 mismatch in {path}: expected {expected_macro_f1}, found {value}.",
    )
    return payload


def validate_benchmark_table(path: Path, *, expected_mean_macro_f1: float) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"Benchmark comparison table has no header: {path}")
        required_columns = {"model_id", "threshold", "macro_f1", "row_complete"}
        missing = sorted(required_columns - set(reader.fieldnames))
        require(not missing, f"Missing required columns in benchmark comparison table {path}: {missing}")

        rows = list(reader)

    require(len(rows) == len(EXPECTED_BENCHMARK_MODELS), f"Expected 5 benchmark rows in {path}.")

    seen_models: set[str] = set()
    thresholds: set[float] = set()
    macro_f1_values: list[float] = []
    for row in rows:
        model_id = str(row["model_id"]).strip()
        require(model_id != "", f"Blank model_id in benchmark comparison table: {path}")
        require(model_id not in seen_models, f"Duplicate model_id {model_id!r} in {path}")
        seen_models.add(model_id)

        threshold = as_float(row["threshold"], label=f"threshold in {path}")
        thresholds.add(threshold)
        macro_f1_values.append(as_float(row["macro_f1"], label=f"macro_f1 in {path}"))
        require(coerce_bool(row["row_complete"]), f"Incomplete benchmark row for {model_id!r} in {path}")

    require(
        seen_models == set(EXPECTED_BENCHMARK_MODELS),
        f"Unexpected benchmark model set in {path}: {sorted(seen_models)}",
    )
    require(thresholds == {EXPECTED_THRESHOLD}, f"Unexpected threshold set in {path}: {sorted(thresholds)}")

    mean_macro_f1 = sum(macro_f1_values) / len(macro_f1_values)
    require(
        nearly_equal(mean_macro_f1, expected_mean_macro_f1, tolerance=1e-9),
        f"Benchmark mean macro-F1 mismatch in {path}: expected {expected_mean_macro_f1}, found {mean_macro_f1}.",
    )

    return {
        "row_count": len(rows),
        "mean_macro_f1": mean_macro_f1,
        "models": sorted(seen_models),
        "threshold": EXPECTED_THRESHOLD,
    }


def format_metric(value: float) -> str:
    return format(value, ".17g")


def make_markdown(
    *,
    generated_at_utc: str,
    manifest_path: Path,
    decision_json_path: Path,
    output_root: Path,
    baseline_label_csv: Path,
    judge_label_csv: Path,
    judge_label_jsonl: Path,
    judge_labeling_report_json: Path,
    judge_labeling_errors_jsonl: Path,
    judge_labeling_run_log: Path,
    judge_gold_metrics_json: Path,
    baseline_gold_metrics_json: Path,
    benchmark_table_csv: Path,
    decision: Mapping[str, Any],
    judge_gold_metrics: Mapping[str, Any],
    hash_count: int,
    warnings: Sequence[str],
    baseline_csv_rows: int,
    judge_csv_rows: int,
    judge_jsonl_rows: int,
    baseline_judge_ids_match: bool,
    judge_csv_jsonl_ids_match: bool,
) -> str:
    benchmark_integrity = expect_mapping(
        decision.get("benchmark_integrity"),
        label="benchmark_integrity in decision summary",
    )
    label_drift_summary = expect_mapping(
        decision.get("label_drift_summary"),
        label="label_drift_summary in decision summary",
    )
    baseline_merge = expect_mapping(
        label_drift_summary.get("baseline_merge_integrity"),
        label="baseline_merge_integrity in decision summary",
    )
    judge_merge = expect_mapping(
        label_drift_summary.get("judge_merge_integrity"),
        label="judge_merge_integrity in decision summary",
    )
    emotion_metrics = expect_mapping(
        judge_gold_metrics.get("emotion_metrics"),
        label="emotion_metrics in judge gold metrics",
    )
    per_label_f1 = expect_mapping(
        emotion_metrics.get("per_label_f1"),
        label="per_label_f1 in judge gold metrics",
    )

    baseline_gold = as_float(decision.get("baseline_gold_macro_f1"), label="baseline_gold_macro_f1")
    judge_gold = as_float(decision.get("judge_gold_macro_f1"), label="judge_gold_macro_f1")
    baseline_benchmark = as_float(
        decision.get("baseline_benchmark_mean_macro_f1"),
        label="baseline_benchmark_mean_macro_f1",
    )
    judge_benchmark = as_float(
        decision.get("judge_benchmark_mean_macro_f1"),
        label="judge_benchmark_mean_macro_f1",
    )

    warnings_line = "None." if not warnings else "; ".join(warnings)
    branch_config = PROJECT_ROOT / IMPLEMENTATION_FILES[0]
    branch_runner = PROJECT_ROOT / IMPLEMENTATION_FILES[1]
    gold_compare = PROJECT_ROOT / IMPLEMENTATION_FILES[2]
    branch_compare = PROJECT_ROOT / IMPLEMENTATION_FILES[3]
    run_script = PROJECT_ROOT / IMPLEMENTATION_FILES[4]

    return f"""# Phase 1 Gemini 2.5 Pro Judge Decision

Freeze generated (UTC): `{generated_at_utc}`

## Project Context

This repo has a frozen 784-perfume baseline built with Gemini 2.5 Flash labels and downstream benchmark artifacts. The canonical frozen baseline label artifact remains `labels/gemini_labels_784_v1.csv` and must stay immutable.

## Why Phase 1 Existed

Phase 1 was a controlled judge-branch comparison to decide whether the future expansion path should keep the frozen Gemini 2.5 Flash labeling policy or switch to Gemini 2.5 Pro. The branch design held the dataset, raw review and transcript universe, snippet selection policy, strict prompt and schema, aggregation rule, folds, seeds, and benchmark protocol fixed. The only intended variable was the judge model.

## Branch Definitions

- Frozen baseline branch: same 784 perfumes, frozen labels from Gemini 2.5 Flash, canonical label CSV `labels/gemini_labels_784_v1.csv`.
- Gemini Pro judge branch: same 784 perfumes and same evaluation contract, but the labeling judge changed to Gemini 2.5 Pro under branch ID `judge_g25pro_v1`.

## Phase 1 Branch Files And Artifacts

Implemented branch files:

- `{stringify_path(branch_config)}`
- `{stringify_path(branch_runner)}`
- `{stringify_path(gold_compare)}`
- `{stringify_path(branch_compare)}`
- `{stringify_path(run_script)}`

Frozen branch artifacts:

- Output root: `{stringify_path(output_root)}`
- Decision summary JSON: `{stringify_path(decision_json_path)}`
- Freeze manifest JSON: `{stringify_path(manifest_path)}`
- Baseline label CSV: `{stringify_path(baseline_label_csv)}`
- Judge label CSV: `{stringify_path(judge_label_csv)}`
- Judge label JSONL: `{stringify_path(judge_label_jsonl)}`
- Judge labeling report JSON: `{stringify_path(judge_labeling_report_json)}`
- Judge labeling errors JSONL: `{stringify_path(judge_labeling_errors_jsonl)}`
- Judge labeling run log: `{stringify_path(judge_labeling_run_log)}`
- Baseline gold metrics JSON: `{stringify_path(baseline_gold_metrics_json)}`
- Judge gold metrics JSON: `{stringify_path(judge_gold_metrics_json)}`
- Judge benchmark comparison table: `{stringify_path(benchmark_table_csv)}`

## Final Decision

`GO`

Gemini 2.5 Pro won Phase 1 under the repo decision rule and should be promoted as the labeling policy candidate for the next expansion stage.

## Key Metrics

| Metric | Frozen baseline: Gemini 2.5 Flash | Judge branch: Gemini 2.5 Pro |
| --- | ---: | ---: |
| Gold macro-F1 | {format_metric(baseline_gold)} | {format_metric(judge_gold)} |
| Benchmark mean macro-F1 | {format_metric(baseline_benchmark)} | {format_metric(judge_benchmark)} |

## Integrity Checks Summary

- Final decision in decision summary: `go`
- `merge_integrity_pass`: `{str(decision.get("merge_integrity_pass")).lower()}`
- `official_collapsed_labels`: `{decision.get("official_collapsed_labels")}`
- Baseline benchmark complete rows: `{benchmark_integrity.get("baseline_complete_rows")}`
- Judge benchmark complete rows: `{benchmark_integrity.get("judge_complete_rows")}`
- Baseline threshold: `{benchmark_integrity.get("baseline_thresholds_complete")}`
- Judge threshold: `{benchmark_integrity.get("judge_thresholds_complete")}`
- Baseline label CSV rows: `{baseline_csv_rows}`
- Judge label CSV rows: `{judge_csv_rows}`
- Judge label JSONL rows: `{judge_jsonl_rows}`
- Judge JSONL metadata model: `{EXPECTED_JUDGE_MODEL}`
- Baseline and judge `perfume_id` sets match: `{str(baseline_judge_ids_match).lower()}`
- Judge CSV and JSONL `perfume_id` sets match: `{str(judge_csv_jsonl_ids_match).lower()}`
- Baseline merge duplicates: `{baseline_merge.get("duplicate_total")}`
- Baseline merge hash mismatches: `{baseline_merge.get("hash_mismatch_count")}`
- Judge merge duplicates: `{judge_merge.get("duplicate_total")}`
- Judge merge hash mismatches: `{judge_merge.get("hash_mismatch_count")}`
- Judge labeling failures: `0`
- Judge missing IDs: `0`
- Key artifact SHA-256 hashes captured in manifest: `{hash_count}`
- Branch outputs remain isolated under `{stringify_path(output_root)}`

## Known Limitations

- Phase 1 answers a branch policy question, not whether the labels are perfect ground truth.
- The matched gold evaluation set remains small at `{EXPECTED_GOLD_ROWS}` perfumes.
- Some official labels remain weak on gold evaluation, especially sadness and surprise. Judge per-label F1 on the gold set was sadness `{format_metric(as_float(per_label_f1.get("sadness"), label="judge sadness F1"))}` and surprise `{format_metric(as_float(per_label_f1.get("surprise"), label="judge surprise F1"))}`.
- Anger was excluded from the primary decision metric and should not be over-interpreted from this Phase 1 decision alone.

## Conclusion

Promote the Gemini 2.5 Pro labeling policy to the next controlled expansion stage. The Phase 1 result is a documented `GO`, with better gold macro-F1 and better benchmark mean macro-F1 than the frozen Gemini 2.5 Flash baseline, while preserving the branch integrity checks required for the 784-perfume comparison.

## Next Phase

Build a versioned 2,000-perfume expansion path using the Gemini 2.5 Pro policy before attempting the later 5,000-perfume scale-up. Do not reuse `judge_g25pro_v1` output roots. Use a new config and output root, likely:

- `config/gemini_judge/judge_g25pro_2k_v1.yaml`
- `outputs/gemini_judge/judge_g25pro_2k_v1/`

Warnings: {warnings_line}
"""


def main() -> None:
    args = parse_args()

    decision_json = resolve_repo_path(args.decision_json).resolve()
    out_json = resolve_repo_path(args.out_json).resolve()
    out_md = resolve_repo_path(args.out_md).resolve()

    output_root = decision_json.parent.parent
    require(output_root.name == EXPECTED_BRANCH_ID, f"Expected branch output root {EXPECTED_BRANCH_ID}, found {output_root.name!r}.")
    require(output_root.is_dir(), f"Branch output root not found: {output_root}")

    decision = load_json(decision_json)
    decision_value = str(decision.get("decision", "")).strip().lower()
    require(decision_value == "go", f"Phase 1 decision must be 'go', found {decision_value!r}.")
    require(coerce_bool(decision.get("merge_integrity_pass")), "merge_integrity_pass must be true.")
    require(
        decision.get("official_collapsed_labels") == [],
        f"official_collapsed_labels must be empty, found {decision.get('official_collapsed_labels')!r}.",
    )

    baseline_gold_macro_f1 = as_float(decision.get("baseline_gold_macro_f1"), label="baseline_gold_macro_f1")
    judge_gold_macro_f1 = as_float(decision.get("judge_gold_macro_f1"), label="judge_gold_macro_f1")
    baseline_benchmark_mean_macro_f1 = as_float(
        decision.get("baseline_benchmark_mean_macro_f1"),
        label="baseline_benchmark_mean_macro_f1",
    )
    judge_benchmark_mean_macro_f1 = as_float(
        decision.get("judge_benchmark_mean_macro_f1"),
        label="judge_benchmark_mean_macro_f1",
    )
    require(
        judge_gold_macro_f1 > baseline_gold_macro_f1,
        "Judge gold macro-F1 must exceed the frozen baseline gold macro-F1 for a GO decision.",
    )
    require(
        judge_benchmark_mean_macro_f1 > baseline_benchmark_mean_macro_f1,
        "Judge benchmark mean macro-F1 must exceed the frozen baseline benchmark mean macro-F1 for a GO decision.",
    )

    benchmark_integrity = expect_mapping(decision.get("benchmark_integrity"), label="benchmark_integrity")
    require(
        as_int(benchmark_integrity.get("baseline_complete_rows"), label="baseline_complete_rows") == 5,
        "Expected 5 complete frozen baseline benchmark rows.",
    )
    require(
        as_int(benchmark_integrity.get("judge_complete_rows"), label="judge_complete_rows") == 5,
        "Expected 5 complete judge benchmark rows.",
    )
    require(
        benchmark_integrity.get("baseline_thresholds_complete") == [EXPECTED_THRESHOLD],
        f"Expected frozen baseline threshold [{EXPECTED_THRESHOLD}].",
    )
    require(
        benchmark_integrity.get("judge_thresholds_complete") == [EXPECTED_THRESHOLD],
        f"Expected judge threshold [{EXPECTED_THRESHOLD}].",
    )
    require(
        coerce_bool(benchmark_integrity.get("baseline_integrity_pass")),
        "baseline_integrity_pass must be true.",
    )
    require(
        coerce_bool(benchmark_integrity.get("judge_integrity_pass")),
        "judge_integrity_pass must be true.",
    )
    require(
        nearly_equal(
            as_float(benchmark_integrity.get("baseline_mean_macro_f1"), label="baseline_mean_macro_f1"),
            baseline_benchmark_mean_macro_f1,
        ),
        "baseline_mean_macro_f1 mismatch between decision summary sections.",
    )
    require(
        nearly_equal(
            as_float(benchmark_integrity.get("judge_mean_macro_f1"), label="judge_mean_macro_f1"),
            judge_benchmark_mean_macro_f1,
        ),
        "judge_mean_macro_f1 mismatch between decision summary sections.",
    )

    label_drift_summary = expect_mapping(decision.get("label_drift_summary"), label="label_drift_summary")
    require(
        as_int(label_drift_summary.get("n_rows"), label="label_drift_summary.n_rows") == EXPECTED_ROW_COUNT,
        f"Expected label drift comparison to cover {EXPECTED_ROW_COUNT} rows.",
    )

    for merge_key in ("baseline_merge_integrity", "judge_merge_integrity"):
        merge_payload = expect_mapping(label_drift_summary.get(merge_key), label=merge_key)
        require(coerce_bool(merge_payload.get("passed")), f"{merge_key}.passed must be true.")
        require(
            as_int(merge_payload.get("matched"), label=f"{merge_key}.matched") == EXPECTED_ROW_COUNT,
            f"{merge_key}.matched must equal {EXPECTED_ROW_COUNT}.",
        )
        require(
            as_int(merge_payload.get("unified_only"), label=f"{merge_key}.unified_only") == 0,
            f"{merge_key}.unified_only must be 0.",
        )
        require(
            as_int(merge_payload.get("gemini_only"), label=f"{merge_key}.gemini_only") == 0,
            f"{merge_key}.gemini_only must be 0.",
        )
        require(
            as_int(merge_payload.get("duplicate_total"), label=f"{merge_key}.duplicate_total") == 0,
            f"{merge_key}.duplicate_total must be 0.",
        )
        require(
            as_int(merge_payload.get("hash_mismatch_count"), label=f"{merge_key}.hash_mismatch_count") == 0,
            f"{merge_key}.hash_mismatch_count must be 0.",
        )

    baseline_label_csv = resolve_repo_path(str(label_drift_summary.get("baseline_label_csv", ""))).resolve()
    judge_label_csv = resolve_repo_path(str(label_drift_summary.get("judge_label_csv", ""))).resolve()
    require(
        stringify_path(baseline_label_csv) == "labels/gemini_labels_784_v1.csv",
        f"Unexpected baseline label CSV path: {stringify_path(baseline_label_csv)}",
    )
    require(
        stringify_path(judge_label_csv) == "labels/gemini_labels_784_judge_g25pro_v1.csv",
        f"Unexpected judge label CSV path: {stringify_path(judge_label_csv)}",
    )

    judge_label_jsonl = PROJECT_ROOT / "labels" / "gemini_labels_784_judge_g25pro_v1.jsonl"
    judge_labeling_report_json = PROJECT_ROOT / "labels" / "gemini_labeling_report_judge_g25pro_v1.json"
    judge_labeling_errors_jsonl = PROJECT_ROOT / "labels" / "gemini_labeling_errors_judge_g25pro_v1.jsonl"
    judge_labeling_run_log = PROJECT_ROOT / "labels" / "gemini_labeling_run_judge_g25pro_v1.log"
    judge_gold_metrics_json = output_root / "gold_eval" / "judge_vs_gold_metrics.json"
    baseline_gold_metrics_json = output_root / "gold_eval" / "baseline_vs_gold_metrics.json"
    benchmark_table_csv = output_root / "model_benchmark" / "tables" / "comparison_table.csv"
    branch_config_yaml = PROJECT_ROOT / "config" / "gemini_judge" / f"{EXPECTED_BRANCH_ID}.yaml"
    branch_runner_py = PROJECT_ROOT / "src" / "pipeline" / "run_gemini_judge_branch.py"
    gold_compare_py = PROJECT_ROOT / "src" / "analysis" / "compare_label_branch_to_gold43.py"
    branch_compare_py = PROJECT_ROOT / "src" / "analysis" / "compare_gemini_branches.py"
    run_phase1_script = PROJECT_ROOT / "scripts" / "run_phase1_judge.sh"
    judge_merge_report_json = output_root / "analysis_out" / "gemini_unified_join_report.json"

    required_paths = (
        decision_json,
        baseline_label_csv,
        judge_label_csv,
        judge_label_jsonl,
        judge_labeling_report_json,
        judge_labeling_errors_jsonl,
        judge_labeling_run_log,
        judge_gold_metrics_json,
        baseline_gold_metrics_json,
        benchmark_table_csv,
        branch_config_yaml,
        branch_runner_py,
        gold_compare_py,
        branch_compare_py,
        run_phase1_script,
        judge_merge_report_json,
    )
    for path in required_paths:
        require(path.exists(), f"Required artifact does not exist: {path}")

    baseline_csv_rows, baseline_ids = collect_csv_perfume_ids(baseline_label_csv)
    judge_csv_rows, judge_csv_ids = collect_csv_perfume_ids(judge_label_csv)
    judge_jsonl_rows, judge_jsonl_ids = collect_jsonl_perfume_ids(
        judge_label_jsonl,
        expected_model=EXPECTED_JUDGE_MODEL,
    )
    require(
        baseline_csv_rows == EXPECTED_ROW_COUNT,
        f"Expected {EXPECTED_ROW_COUNT} baseline CSV rows, found {baseline_csv_rows}.",
    )
    require(
        judge_csv_rows == EXPECTED_ROW_COUNT,
        f"Expected {EXPECTED_ROW_COUNT} judge CSV rows, found {judge_csv_rows}.",
    )
    require(
        judge_jsonl_rows == EXPECTED_ROW_COUNT,
        f"Expected {EXPECTED_ROW_COUNT} judge JSONL rows, found {judge_jsonl_rows}.",
    )
    require(
        baseline_ids == judge_csv_ids,
        "Frozen baseline CSV and judge CSV perfume_id sets do not match.",
    )
    require(
        judge_csv_ids == judge_jsonl_ids,
        "Judge CSV and judge JSONL perfume_id sets do not match.",
    )

    labeling_report = validate_labeling_report(judge_labeling_report_json)
    baseline_gold_metrics = validate_gold_metrics(
        baseline_gold_metrics_json,
        expected_branch_name="baseline",
        expected_macro_f1=baseline_gold_macro_f1,
    )
    judge_gold_metrics = validate_gold_metrics(
        judge_gold_metrics_json,
        expected_branch_name="judge",
        expected_macro_f1=judge_gold_macro_f1,
    )
    benchmark_summary = validate_benchmark_table(
        benchmark_table_csv,
        expected_mean_macro_f1=judge_benchmark_mean_macro_f1,
    )

    file_hash_paths = (
        branch_config_yaml,
        branch_runner_py,
        gold_compare_py,
        branch_compare_py,
        run_phase1_script,
        baseline_label_csv,
        judge_label_csv,
        judge_label_jsonl,
        judge_labeling_report_json,
        judge_labeling_errors_jsonl,
        judge_labeling_run_log,
        judge_merge_report_json,
        baseline_gold_metrics_json,
        judge_gold_metrics_json,
        benchmark_table_csv,
        decision_json,
    )
    file_hashes_sha256 = {
        stringify_path(path): sha256_file(path)
        for path in file_hash_paths
    }

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    warnings: list[str] = []

    manifest = {
        "branch_id": EXPECTED_BRANCH_ID,
        "judge_model": EXPECTED_JUDGE_MODEL,
        "baseline_label_csv": stringify_path(baseline_label_csv),
        "judge_label_csv": stringify_path(judge_label_csv),
        "judge_label_jsonl": stringify_path(judge_label_jsonl),
        "decision_summary_json": stringify_path(decision_json),
        "gold_metrics_json": stringify_path(judge_gold_metrics_json),
        "benchmark_table_csv": stringify_path(benchmark_table_csv),
        "output_root": stringify_path(output_root),
        "final_decision": decision_value,
        "baseline_gold_macro_f1": baseline_gold_macro_f1,
        "judge_gold_macro_f1": judge_gold_macro_f1,
        "baseline_benchmark_mean_macro_f1": baseline_benchmark_mean_macro_f1,
        "judge_benchmark_mean_macro_f1": judge_benchmark_mean_macro_f1,
        "merge_integrity_pass": True,
        "official_collapsed_labels": [],
        "timestamp_utc": generated_at_utc,
        "implementation_files": list(IMPLEMENTATION_FILES),
        "row_counts": {
            "baseline_label_csv_rows": baseline_csv_rows,
            "judge_label_csv_rows": judge_csv_rows,
            "judge_label_jsonl_rows": judge_jsonl_rows,
        },
        "integrity_summary": {
            "baseline_and_judge_perfume_ids_match": True,
            "judge_csv_and_jsonl_perfume_ids_match": True,
            "judge_jsonl_metadata_model": EXPECTED_JUDGE_MODEL,
            "judge_labeling_failed_count": as_int(
                expect_mapping(labeling_report.get("counts"), label="labeling_report.counts").get("failed"),
                label="labeling report failed count",
            ),
            "judge_labeling_missing_ids_count": len(labeling_report.get("missing_ids", [])),
            "baseline_benchmark_complete_rows": as_int(
                benchmark_integrity.get("baseline_complete_rows"),
                label="baseline_complete_rows",
            ),
            "judge_benchmark_complete_rows": benchmark_summary["row_count"],
            "baseline_thresholds_complete": benchmark_integrity.get("baseline_thresholds_complete"),
            "judge_thresholds_complete": [benchmark_summary["threshold"]],
            "baseline_merge_duplicate_total": as_int(
                expect_mapping(
                    label_drift_summary.get("baseline_merge_integrity"),
                    label="baseline_merge_integrity",
                ).get("duplicate_total"),
                label="baseline_merge_integrity.duplicate_total",
            ),
            "judge_merge_duplicate_total": as_int(
                expect_mapping(
                    label_drift_summary.get("judge_merge_integrity"),
                    label="judge_merge_integrity",
                ).get("duplicate_total"),
                label="judge_merge_integrity.duplicate_total",
            ),
            "baseline_merge_hash_mismatch_count": as_int(
                expect_mapping(
                    label_drift_summary.get("baseline_merge_integrity"),
                    label="baseline_merge_integrity",
                ).get("hash_mismatch_count"),
                label="baseline_merge_integrity.hash_mismatch_count",
            ),
            "judge_merge_hash_mismatch_count": as_int(
                expect_mapping(
                    label_drift_summary.get("judge_merge_integrity"),
                    label="judge_merge_integrity",
                ).get("hash_mismatch_count"),
                label="judge_merge_integrity.hash_mismatch_count",
            ),
        },
        "limitations": {
            "matched_gold_rows": EXPECTED_GOLD_ROWS,
            "primary_decision_metric_excluded_labels": list(EXPECTED_PRIMARY_EXCLUDED_LABELS),
            "judge_gold_per_label_f1": expect_mapping(
                expect_mapping(judge_gold_metrics.get("emotion_metrics"), label="judge emotion_metrics").get("per_label_f1"),
                label="judge per_label_f1",
            ),
        },
        "related_artifacts": {
            "baseline_gold_metrics_json": stringify_path(baseline_gold_metrics_json),
            "judge_labeling_report_json": stringify_path(judge_labeling_report_json),
            "judge_labeling_errors_jsonl": stringify_path(judge_labeling_errors_jsonl),
            "judge_labeling_run_log": stringify_path(judge_labeling_run_log),
            "judge_merge_report_json": stringify_path(judge_merge_report_json),
        },
        "artifact_hash_count": len(file_hashes_sha256),
        "file_hashes_sha256": file_hashes_sha256,
        "warnings": warnings,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(
        make_markdown(
            generated_at_utc=generated_at_utc,
            manifest_path=out_json,
            decision_json_path=decision_json,
            output_root=output_root,
            baseline_label_csv=baseline_label_csv,
            judge_label_csv=judge_label_csv,
            judge_label_jsonl=judge_label_jsonl,
            judge_labeling_report_json=judge_labeling_report_json,
            judge_labeling_errors_jsonl=judge_labeling_errors_jsonl,
            judge_labeling_run_log=judge_labeling_run_log,
            judge_gold_metrics_json=judge_gold_metrics_json,
            baseline_gold_metrics_json=baseline_gold_metrics_json,
            benchmark_table_csv=benchmark_table_csv,
            decision=decision,
            judge_gold_metrics=judge_gold_metrics,
            hash_count=len(file_hashes_sha256),
            warnings=warnings,
            baseline_csv_rows=baseline_csv_rows,
            judge_csv_rows=judge_csv_rows,
            judge_jsonl_rows=judge_jsonl_rows,
            baseline_judge_ids_match=True,
            judge_csv_jsonl_ids_match=True,
        ),
        encoding="utf-8",
    )

    print("Phase 1 freeze complete")
    print(f"Decision: {decision_value.upper()}")
    print(f"Baseline gold macro-F1: {format_metric(baseline_gold_macro_f1)}")
    print(f"Judge gold macro-F1: {format_metric(judge_gold_macro_f1)}")
    print(f"Baseline benchmark mean macro-F1: {format_metric(baseline_benchmark_mean_macro_f1)}")
    print(f"Judge benchmark mean macro-F1: {format_metric(judge_benchmark_mean_macro_f1)}")
    print(f"Manifest: {stringify_path(out_json)}")
    print(f"Markdown note: {stringify_path(out_md)}")
    print(f"Artifact hashes: {len(file_hashes_sha256)}")
    print("Warnings: none")


if __name__ == "__main__":
    main()
