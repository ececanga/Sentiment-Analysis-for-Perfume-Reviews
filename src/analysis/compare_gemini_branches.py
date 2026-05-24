# Date Created: 2026-05-20

"""Compare the frozen Gemini baseline branch against a Gemini judge branch and issue a Phase 1 decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.labeling.strict_schema import EMOTIONS
from src.utils.paths import PROJECT_ROOT


VALID_DECISIONS = {"go", "no_go", "inconclusive"}


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-label-csv",
        type=Path,
        default=PROJECT_ROOT / "labels" / "gemini_labels_784_v1.csv",
        help="Canonical frozen baseline label CSV.",
    )
    parser.add_argument(
        "--judge-label-csv",
        type=Path,
        default=PROJECT_ROOT / "labels" / "gemini_labels_784_judge_g25pro_v1.csv",
        help="Judge branch label CSV.",
    )
    parser.add_argument(
        "--baseline-merge-report-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis_out" / "gemini_unified_join_report.json",
        help="Frozen baseline merge integrity report.",
    )
    parser.add_argument(
        "--judge-merge-report-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gemini_judge" / "judge_g25pro_v1" / "analysis_out" / "gemini_unified_join_report.json",
        help="Judge branch merge integrity report.",
    )
    parser.add_argument(
        "--baseline-benchmark-comparison-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model_benchmark" / "tables" / "comparison_table.csv",
        help="Frozen baseline benchmark comparison table.",
    )
    parser.add_argument(
        "--judge-benchmark-comparison-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gemini_judge" / "judge_g25pro_v1" / "model_benchmark" / "tables" / "comparison_table.csv",
        help="Judge branch benchmark comparison table.",
    )
    parser.add_argument(
        "--baseline-gold-metrics-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gemini_judge" / "judge_g25pro_v1" / "gold_eval" / "baseline_vs_gold_metrics.json",
        help="Baseline-vs-gold metrics JSON from the branch gold comparison step.",
    )
    parser.add_argument(
        "--judge-gold-metrics-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gemini_judge" / "judge_g25pro_v1" / "gold_eval" / "judge_vs_gold_metrics.json",
        help="Judge-vs-gold metrics JSON from the branch gold comparison step.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "evaluation_config.yaml",
        help="Official frozen evaluation config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gemini_judge" / "judge_g25pro_v1" / "decision",
        help="Output directory for final branch comparison artifacts.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML payload: {path}")
    return payload


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload: {path}")
    return payload


def require_columns(df: pd.DataFrame, columns: Iterable[str], *, path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_label_frame(path: Path, *, all_emotions: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Label CSV not found: {path}")
    df = pd.read_csv(path)
    require_columns(df, ["perfume_id", *all_emotions], path=path)
    df["perfume_id"] = df["perfume_id"].fillna("").astype(str).str.strip()
    if df["perfume_id"].eq("").any():
        raise ValueError(f"Blank perfume_id detected in {path}")
    dup_ids = sorted(df.loc[df["perfume_id"].duplicated(keep=False), "perfume_id"].astype(str).unique().tolist())
    if dup_ids:
        raise ValueError(f"Duplicate perfume_id values in {path}: {dup_ids[:10]}")
    for label in all_emotions:
        df[label] = pd.to_numeric(df[label], errors="coerce")
        invalid = df[label].isna() | ~df[label].isin([0, 1])
        if invalid.any():
            raise ValueError(f"Invalid binary values in {path} for column {label}")
    return df.sort_values("perfume_id").reset_index(drop=True)


def compute_prevalence_table(
    *,
    baseline_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    official_labels: Sequence[str],
) -> pd.DataFrame:
    baseline_ids = set(baseline_df["perfume_id"].astype(str))
    judge_ids = set(judge_df["perfume_id"].astype(str))
    if baseline_ids != judge_ids:
        missing_from_judge = sorted(list(baseline_ids - judge_ids))[:10]
        missing_from_baseline = sorted(list(judge_ids - baseline_ids))[:10]
        raise ValueError(
            "Baseline and judge perfume_id sets do not match. "
            f"Missing from judge={missing_from_judge} missing from baseline={missing_from_baseline}"
        )

    rows: List[Dict[str, Any]] = []
    for label in EMOTIONS:
        if label not in baseline_df.columns or label not in judge_df.columns:
            continue
        baseline_vals = baseline_df[label].to_numpy(dtype=int)
        judge_vals = judge_df[label].to_numpy(dtype=int)
        baseline_support = int(baseline_vals.sum())
        judge_support = int(judge_vals.sum())
        rows.append(
            {
                "label": label,
                "in_official_target_set": label in set(official_labels),
                "baseline_support": baseline_support,
                "judge_support": judge_support,
                "support_delta": int(judge_support - baseline_support),
                "baseline_prevalence": float(baseline_vals.mean()),
                "judge_prevalence": float(judge_vals.mean()),
                "prevalence_delta": float(judge_vals.mean() - baseline_vals.mean()),
                "collapsed_to_zero": bool(label in set(official_labels) and baseline_support > 0 and judge_support == 0),
            }
        )
    return pd.DataFrame(rows)


def normalize_merge_report_counts(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        counts = payload.get("join_counts")
    integrity = payload.get("integrity", {})
    if not isinstance(counts, Mapping) or not isinstance(integrity, Mapping):
        raise ValueError("Merge report is missing counts/integrity sections.")
    return counts, integrity


def assess_merge_integrity(payload: Mapping[str, Any], *, expected_rows: int) -> Dict[str, Any]:
    counts, integrity = normalize_merge_report_counts(payload)
    matched = int(counts.get("matched", -1))
    unified_only = int(counts.get("unified_only", -1))
    gemini_only = int(counts.get("gemini_only", -1))
    duplicate_total = int(integrity.get("duplicate_ids_unified_count", 0)) + int(
        integrity.get("duplicate_ids_gemini_count", 0)
    )
    hash_mismatch_count = int(integrity.get("hash_mismatch_count", -1))
    passed = (
        matched == expected_rows
        and unified_only == 0
        and gemini_only == 0
        and duplicate_total == 0
        and hash_mismatch_count == 0
    )
    return {
        "passed": bool(passed),
        "matched": matched,
        "expected_rows": int(expected_rows),
        "unified_only": unified_only,
        "gemini_only": gemini_only,
        "duplicate_total": duplicate_total,
        "hash_mismatch_count": hash_mismatch_count,
    }


def load_benchmark_table(
    *,
    path: Path,
    expected_models: Sequence[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark comparison table not found: {path}")
    df = pd.read_csv(path)
    require_columns(df, ["model_id", "threshold", "macro_f1", "row_complete"], path=path)
    df = df.loc[df["model_id"].isin(expected_models)].copy()
    if df["model_id"].duplicated().any():
        dup_ids = sorted(df.loc[df["model_id"].duplicated(keep=False), "model_id"].astype(str).unique().tolist())
        raise ValueError(f"Duplicate model rows in benchmark comparison table {path}: {dup_ids}")
    for column in ["threshold", "macro_f1"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["row_complete"] = df["row_complete"].map(parse_bool)
    return df.sort_values("model_id").reset_index(drop=True)


def build_benchmark_side_by_side(
    *,
    baseline_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    expected_models: Sequence[str],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    baseline_complete = 0
    judge_complete = 0
    baseline_values: List[float] = []
    judge_values: List[float] = []

    for model_id in expected_models:
        base_row = baseline_df.loc[baseline_df["model_id"] == model_id]
        judge_row = judge_df.loc[judge_df["model_id"] == model_id]
        if base_row.empty or judge_row.empty:
            rows.append(
                {
                    "model_id": model_id,
                    "baseline_threshold": np.nan,
                    "judge_threshold": np.nan,
                    "baseline_row_complete": False,
                    "judge_row_complete": False,
                    "baseline_macro_f1": np.nan,
                    "judge_macro_f1": np.nan,
                    "delta_macro_f1": np.nan,
                }
            )
            continue

        base = base_row.iloc[0]
        judge = judge_row.iloc[0]
        base_complete = bool(base["row_complete"])
        judge_complete_flag = bool(judge["row_complete"])
        if base_complete:
            baseline_complete += 1
        if judge_complete_flag:
            judge_complete += 1

        base_macro = float(base["macro_f1"]) if pd.notna(base["macro_f1"]) else float("nan")
        judge_macro = float(judge["macro_f1"]) if pd.notna(judge["macro_f1"]) else float("nan")
        if base_complete and pd.notna(base_macro):
            baseline_values.append(base_macro)
        if judge_complete_flag and pd.notna(judge_macro):
            judge_values.append(judge_macro)

        rows.append(
            {
                "model_id": model_id,
                "baseline_threshold": float(base["threshold"]) if pd.notna(base["threshold"]) else float("nan"),
                "judge_threshold": float(judge["threshold"]) if pd.notna(judge["threshold"]) else float("nan"),
                "baseline_row_complete": base_complete,
                "judge_row_complete": judge_complete_flag,
                "baseline_macro_f1": base_macro,
                "judge_macro_f1": judge_macro,
                "delta_macro_f1": judge_macro - base_macro if pd.notna(base_macro) and pd.notna(judge_macro) else float("nan"),
            }
        )

    baseline_mean = float(np.mean(baseline_values)) if baseline_values else float("nan")
    judge_mean = float(np.mean(judge_values)) if judge_values else float("nan")
    rows.append(
        {
            "model_id": "__mean__",
            "baseline_threshold": float("nan"),
            "judge_threshold": float("nan"),
            "baseline_row_complete": baseline_complete == len(expected_models),
            "judge_row_complete": judge_complete == len(expected_models),
            "baseline_macro_f1": baseline_mean,
            "judge_macro_f1": judge_mean,
            "delta_macro_f1": judge_mean - baseline_mean if pd.notna(baseline_mean) and pd.notna(judge_mean) else float("nan"),
        }
    )

    baseline_thresholds = sorted(
        {
            round(float(value), 6)
            for value in baseline_df.loc[baseline_df["row_complete"], "threshold"].dropna().tolist()
        }
    )
    judge_thresholds = sorted(
        {
            round(float(value), 6)
            for value in judge_df.loc[judge_df["row_complete"], "threshold"].dropna().tolist()
        }
    )
    summary = {
        "expected_models": list(expected_models),
        "baseline_complete_rows": int(baseline_complete),
        "judge_complete_rows": int(judge_complete),
        "baseline_mean_macro_f1": baseline_mean,
        "judge_mean_macro_f1": judge_mean,
        "baseline_thresholds_complete": baseline_thresholds,
        "judge_thresholds_complete": judge_thresholds,
        "baseline_integrity_pass": baseline_complete == len(expected_models) and baseline_thresholds == [0.3],
        "judge_integrity_pass": judge_complete == len(expected_models) and judge_thresholds == [0.3],
    }
    return pd.DataFrame(rows), summary


def build_gold_side_by_side(
    *,
    baseline_payload: Mapping[str, Any],
    judge_payload: Mapping[str, Any],
    all_emotions: Sequence[str],
) -> pd.DataFrame:
    rows = [
        {
            "metric": "primary_macro_f1_official",
            "baseline_value": float(baseline_payload["primary_decision_metric"]["value"]),
            "judge_value": float(judge_payload["primary_decision_metric"]["value"]),
        },
        {
            "metric": "micro_f1_official",
            "baseline_value": float(baseline_payload["emotion_metrics"]["micro_f1_official"]),
            "judge_value": float(judge_payload["emotion_metrics"]["micro_f1_official"]),
        },
        {
            "metric": "subset_accuracy_official",
            "baseline_value": float(baseline_payload["emotion_metrics"]["subset_accuracy_official"]),
            "judge_value": float(judge_payload["emotion_metrics"]["subset_accuracy_official"]),
        },
        {
            "metric": "vad_mae_average",
            "baseline_value": float(baseline_payload["vad_metrics"]["average"]["mae"]),
            "judge_value": float(judge_payload["vad_metrics"]["average"]["mae"]),
        },
        {
            "metric": "vad_spearman_average",
            "baseline_value": float(baseline_payload["vad_metrics"]["average"]["spearman_r"]),
            "judge_value": float(judge_payload["vad_metrics"]["average"]["spearman_r"]),
        },
    ]
    for label in all_emotions:
        rows.append(
            {
                "metric": f"f1_{label}",
                "baseline_value": float(baseline_payload["emotion_metrics"]["per_label_f1"][label]),
                "judge_value": float(judge_payload["emotion_metrics"]["per_label_f1"][label]),
            }
        )

    gold_df = pd.DataFrame(rows)
    gold_df["delta"] = gold_df["judge_value"] - gold_df["baseline_value"]
    return gold_df


def decide_outcome(
    *,
    merge_integrity_pass: bool,
    official_collapsed_labels: Sequence[str],
    baseline_gold_macro_f1: float,
    judge_gold_macro_f1: float,
    baseline_benchmark_mean_macro_f1: float,
    judge_benchmark_mean_macro_f1: float,
) -> tuple[str, List[str]]:
    reasons: List[str] = []
    if not merge_integrity_pass:
        reasons.append("merge_integrity_failed")
        return "no_go", reasons
    if official_collapsed_labels:
        reasons.append("official_label_collapsed_to_zero")
        return "no_go", reasons
    if judge_gold_macro_f1 < baseline_gold_macro_f1:
        reasons.append("judge_gold_macro_f1_worse_than_baseline")
        return "no_go", reasons
    if judge_gold_macro_f1 > baseline_gold_macro_f1 and judge_benchmark_mean_macro_f1 > baseline_benchmark_mean_macro_f1:
        reasons.append("judge_gold_macro_f1_improved")
        reasons.append("judge_benchmark_mean_macro_f1_improved")
        return "go", reasons
    reasons.append("mixed_or_non_improving_signals")
    return "inconclusive", reasons


def run_comparison(
    *,
    baseline_label_csv: Path,
    judge_label_csv: Path,
    baseline_merge_report_json: Path,
    judge_merge_report_json: Path,
    baseline_benchmark_comparison_csv: Path,
    judge_benchmark_comparison_csv: Path,
    baseline_gold_metrics_json: Path,
    judge_gold_metrics_json: Path,
    config_path: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(config_path)
    official_labels = list(config["targets"]["official_emotion_columns"])
    benchmark_models = list(config["models"]["frozen_model_set"])
    all_emotions = [label for label in EMOTIONS if label in official_labels or label in {"anger"}]

    baseline_df = load_label_frame(baseline_label_csv, all_emotions=all_emotions)
    judge_df = load_label_frame(judge_label_csv, all_emotions=all_emotions)
    prevalence_df = compute_prevalence_table(
        baseline_df=baseline_df,
        judge_df=judge_df,
        official_labels=official_labels,
    )

    baseline_merge_payload = load_json(baseline_merge_report_json)
    judge_merge_payload = load_json(judge_merge_report_json)
    baseline_merge_integrity = assess_merge_integrity(baseline_merge_payload, expected_rows=len(baseline_df))
    judge_merge_integrity = assess_merge_integrity(judge_merge_payload, expected_rows=len(judge_df))

    baseline_gold_payload = load_json(baseline_gold_metrics_json)
    judge_gold_payload = load_json(judge_gold_metrics_json)
    baseline_benchmark_df = load_benchmark_table(
        path=baseline_benchmark_comparison_csv,
        expected_models=benchmark_models,
    )
    judge_benchmark_df = load_benchmark_table(
        path=judge_benchmark_comparison_csv,
        expected_models=benchmark_models,
    )
    benchmark_side_by_side_df, benchmark_summary = build_benchmark_side_by_side(
        baseline_df=baseline_benchmark_df,
        judge_df=judge_benchmark_df,
        expected_models=benchmark_models,
    )
    gold_side_by_side_df = build_gold_side_by_side(
        baseline_payload=baseline_gold_payload,
        judge_payload=judge_gold_payload,
        all_emotions=all_emotions,
    )

    official_collapse_df = prevalence_df.loc[prevalence_df["collapsed_to_zero"]]
    official_collapsed_labels = official_collapse_df["label"].astype(str).tolist()
    merge_integrity_pass = bool(baseline_merge_integrity["passed"] and judge_merge_integrity["passed"])
    decision, reasons = decide_outcome(
        merge_integrity_pass=merge_integrity_pass,
        official_collapsed_labels=official_collapsed_labels,
        baseline_gold_macro_f1=float(baseline_gold_payload["primary_decision_metric"]["value"]),
        judge_gold_macro_f1=float(judge_gold_payload["primary_decision_metric"]["value"]),
        baseline_benchmark_mean_macro_f1=float(benchmark_summary["baseline_mean_macro_f1"]),
        judge_benchmark_mean_macro_f1=float(benchmark_summary["judge_mean_macro_f1"]),
    )
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid decision outcome generated: {decision}")

    drift_summary = {
        "baseline_label_csv": stringify_path(baseline_label_csv),
        "judge_label_csv": stringify_path(judge_label_csv),
        "n_rows": int(len(baseline_df)),
        "max_abs_prevalence_delta": float(prevalence_df["prevalence_delta"].abs().max()) if not prevalence_df.empty else float("nan"),
        "max_abs_prevalence_delta_label": (
            str(prevalence_df.loc[prevalence_df["prevalence_delta"].abs().idxmax(), "label"])
            if not prevalence_df.empty
            else ""
        ),
        "official_collapsed_labels": official_collapsed_labels,
        "baseline_merge_integrity": baseline_merge_integrity,
        "judge_merge_integrity": judge_merge_integrity,
    }

    summary_payload = {
        "decision": decision,
        "decision_reasons": reasons,
        "merge_integrity_pass": merge_integrity_pass,
        "official_collapsed_labels": official_collapsed_labels,
        "baseline_gold_macro_f1": float(baseline_gold_payload["primary_decision_metric"]["value"]),
        "judge_gold_macro_f1": float(judge_gold_payload["primary_decision_metric"]["value"]),
        "baseline_benchmark_mean_macro_f1": float(benchmark_summary["baseline_mean_macro_f1"]),
        "judge_benchmark_mean_macro_f1": float(benchmark_summary["judge_mean_macro_f1"]),
        "benchmark_integrity": benchmark_summary,
        "label_drift_summary": drift_summary,
        "artifacts": {
            "label_prevalence_delta_csv": stringify_path(out_dir / "label_prevalence_delta.csv"),
            "label_drift_summary_json": stringify_path(out_dir / "branch_label_drift_summary.json"),
            "gold_side_by_side_csv": stringify_path(out_dir / "gold_side_by_side.csv"),
            "benchmark_side_by_side_csv": stringify_path(out_dir / "benchmark_side_by_side.csv"),
            "decision_summary_md": stringify_path(out_dir / "branch_decision_summary.md"),
            "decision_summary_json": stringify_path(out_dir / "branch_decision_summary.json"),
        },
    }

    prevalence_df.to_csv(out_dir / "label_prevalence_delta.csv", index=False)
    (out_dir / "branch_label_drift_summary.json").write_text(json.dumps(drift_summary, indent=2), encoding="utf-8")
    gold_side_by_side_df.to_csv(out_dir / "gold_side_by_side.csv", index=False)
    benchmark_side_by_side_df.to_csv(out_dir / "benchmark_side_by_side.csv", index=False)
    (out_dir / "branch_decision_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    md_lines = [
        f"# Phase 1 Judge Branch Decision: {decision}",
        "",
        "## Decision Inputs",
        f"- Merge integrity pass: `{merge_integrity_pass}`",
        f"- Baseline gold macro-F1: `{summary_payload['baseline_gold_macro_f1']:.6f}`",
        f"- Judge gold macro-F1: `{summary_payload['judge_gold_macro_f1']:.6f}`",
        f"- Baseline benchmark mean macro-F1: `{summary_payload['baseline_benchmark_mean_macro_f1']:.6f}`",
        f"- Judge benchmark mean macro-F1: `{summary_payload['judge_benchmark_mean_macro_f1']:.6f}`",
        f"- Official collapsed labels: `{', '.join(official_collapsed_labels) if official_collapsed_labels else 'none'}`",
        "",
        "## Decision Reasons",
        *[f"- {reason}" for reason in reasons],
        "",
        "## Integrity Notes",
        f"- Baseline merge integrity: `{json.dumps(baseline_merge_integrity, sort_keys=True)}`",
        f"- Judge merge integrity: `{json.dumps(judge_merge_integrity, sort_keys=True)}`",
        f"- Benchmark integrity: `{json.dumps(benchmark_summary, sort_keys=True)}`",
    ]
    (out_dir / "branch_decision_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Decision: {decision}")
    print(
        "Gold macro-F1 "
        f"baseline={summary_payload['baseline_gold_macro_f1']:.6f} "
        f"judge={summary_payload['judge_gold_macro_f1']:.6f}"
    )
    print(
        "Benchmark mean macro-F1 "
        f"baseline={summary_payload['baseline_benchmark_mean_macro_f1']:.6f} "
        f"judge={summary_payload['judge_benchmark_mean_macro_f1']:.6f}"
    )
    print(f"Wrote: {out_dir / 'label_prevalence_delta.csv'}")
    print(f"Wrote: {out_dir / 'branch_label_drift_summary.json'}")
    print(f"Wrote: {out_dir / 'gold_side_by_side.csv'}")
    print(f"Wrote: {out_dir / 'benchmark_side_by_side.csv'}")
    print(f"Wrote: {out_dir / 'branch_decision_summary.md'}")
    print(f"Wrote: {out_dir / 'branch_decision_summary.json'}")

    return summary_payload


def main() -> None:
    args = parse_args()
    run_comparison(
        baseline_label_csv=args.baseline_label_csv,
        judge_label_csv=args.judge_label_csv,
        baseline_merge_report_json=args.baseline_merge_report_json,
        judge_merge_report_json=args.judge_merge_report_json,
        baseline_benchmark_comparison_csv=args.baseline_benchmark_comparison_csv,
        judge_benchmark_comparison_csv=args.judge_benchmark_comparison_csv,
        baseline_gold_metrics_json=args.baseline_gold_metrics_json,
        judge_gold_metrics_json=args.judge_gold_metrics_json,
        config_path=args.config,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
