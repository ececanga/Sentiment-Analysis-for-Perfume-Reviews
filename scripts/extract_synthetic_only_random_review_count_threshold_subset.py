#!/usr/bin/env python3
"""Extract one threshold from a synthetic-only random review-count sweep."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs" / "synthetic_only_random_review_count_sweep_gemini_browser_6reviews_thresholds"
DEFAULT_OUT = ROOT / "outputs" / "synthetic_only_random_review_count_sweep_gemini_browser_6reviews_threshold05"

FILES = {
    "runs": "synthetic_only_random_review_count_sweep_runs.csv",
    "summary": "synthetic_only_random_review_count_sweep_summary.csv",
    "per_label": "synthetic_only_random_review_count_sweep_per_label.csv",
}


def stringify(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a focused threshold subset from a threshold sweep.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def format_float(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def write_markdown(path: Path, summary: pd.DataFrame, threshold: float) -> None:
    metric_cols = [
        "model_family",
        "synthetic_reviews",
        "synthetic_perfumes",
        "runs",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_roc_auc_mean",
        "macro_roc_auc_std",
        "macro_pr_auc_mean",
        "macro_pr_auc_std",
    ]
    lines = [
        f"# Randomized Synthetic-Only Review Count Sweep, Threshold {threshold:g}",
        "",
        (
            "Synthetic-only grouped cross-validation with random perfume-order subsets. "
            "Each count is a complete 6-review perfume block."
        ),
        "",
        f"Decision threshold: {threshold:g}",
        "Permutation seeds: 101, 202, 303, 404, 505",
        "",
        "Figures:",
        "",
        "- `figures/synthetic_only_random_review_count_sweep_metrics.svg`",
        "- `figures/synthetic_only_random_review_count_sweep_metrics.png`",
        "- `figures/synthetic_only_random_review_count_sweep_table.svg`",
        "- `figures/synthetic_only_random_review_count_sweep_table.png`",
        "",
        "| " + " | ".join(metric_cols) + " |",
        "| " + " | ".join("---" for _ in metric_cols) + " |",
    ]
    for _, row in summary[metric_cols].iterrows():
        lines.append("| " + " | ".join(format_float(row[col]) for col in metric_cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {}
    for key, filename in FILES.items():
        frame = pd.read_csv(args.source_root / filename)
        if "threshold" not in frame.columns:
            raise ValueError(f"{filename} does not have a threshold column.")
        subset = frame[frame["threshold"].round(6).eq(args.threshold)].copy()
        if subset.empty:
            raise ValueError(f"No rows found for threshold {args.threshold:g} in {filename}.")
        out_path = args.out_root / filename
        subset.to_csv(out_path, index=False)
        frames[key] = subset
        outputs[f"{key}_csv"] = stringify(out_path)

    summary_md = args.out_root / "synthetic_only_random_review_count_sweep_summary.md"
    write_markdown(summary_md, frames["summary"], args.threshold)
    outputs["summary_md"] = stringify(summary_md)

    report_path = args.out_root / "synthetic_only_random_review_count_sweep_report.json"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_threshold_sweep": stringify(args.source_root),
        "threshold": args.threshold,
        "review_counts": sorted(frames["summary"]["synthetic_reviews"].astype(int).unique().tolist()),
        "reviews_per_perfume": 6,
        "permutation_seeds": sorted(frames["runs"]["permutation_seed"].astype(int).unique().tolist()),
        "models": sorted(frames["runs"]["model_id"].unique().tolist()),
        "validation": {
            "runs_rows": int(len(frames["runs"])),
            "summary_rows": int(len(frames["summary"])),
            "per_label_rows": int(len(frames["per_label"])),
            "thresholds": sorted(float(value) for value in frames["runs"]["threshold"].unique().tolist()),
            "max_reviews": int(frames["runs"]["synthetic_reviews"].max()),
            "max_perfumes": int(frames["runs"]["synthetic_perfumes"].max()),
        },
        "outputs": outputs,
        "note": (
            f"Focused threshold-{args.threshold:g} subset extracted from the 784x6 synthetic-only "
            "threshold sensitivity sweep."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    outputs["report_json"] = stringify(report_path)
    print(json.dumps(report["validation"], indent=2))


if __name__ == "__main__":
    main()
