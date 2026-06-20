# Date Created: 2026-06-20
# Date Last Modified: 2026-06-20

"""Combine emotion-rich v2 3-review and reviews4x6 labeled feature files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.labeling.strict_schema import EMOTIONS, VAD_DIMS
from src.synthetic.evaluate_labeled_review_level_pilot import load_labeled_feature_frame
from src.utils.paths import PROJECT_ROOT


EXPECTED_ROWS = 4704
EXPECTED_PERFUMES = 784
EXPECTED_REVIEWS_PER_PERFUME = 6
EXPECTED_REVIEW_INDICES = [1, 2, 3, 4, 5, 6]


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def validate_combined(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(df)}.")
    if df["synthetic_review_id"].nunique() != EXPECTED_ROWS:
        raise ValueError("synthetic_review_id values are not unique.")
    if df["perfume_id"].nunique() != EXPECTED_PERFUMES:
        raise ValueError(f"Expected {EXPECTED_PERFUMES} perfumes, found {df['perfume_id'].nunique()}.")

    review_indices = sorted(pd.to_numeric(df["review_index"], errors="raise").astype(int).unique().tolist())
    if review_indices != EXPECTED_REVIEW_INDICES:
        raise ValueError(f"Unexpected review indices: {review_indices}")

    per_perfume_counts = df.groupby(df["perfume_id"].astype(str)).size()
    bad_counts = per_perfume_counts[per_perfume_counts.ne(EXPECTED_REVIEWS_PER_PERFUME)]
    if not bad_counts.empty:
        preview = ", ".join(f"{perfume_id}={count}" for perfume_id, count in bad_counts.head(10).items())
        raise ValueError(f"Expected {EXPECTED_REVIEWS_PER_PERFUME} rows per perfume. Bad counts: {preview}")

    label_cols = [column for column in EMOTIONS + VAD_DIMS if column in df.columns]
    missing_label_values = int(df[label_cols].isna().sum().sum())
    if missing_label_values:
        raise ValueError(f"Found {missing_label_values} missing label/VAD values.")

    return {
        "rows": int(len(df)),
        "unique_synthetic_review_ids": int(df["synthetic_review_id"].nunique()),
        "unique_perfumes": int(df["perfume_id"].nunique()),
        "review_indices": review_indices,
        "per_perfume_counts": sorted(int(value) for value in per_perfume_counts.unique().tolist()),
        "missing_label_values": missing_label_values,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined 784x6 labeled synthetic review-level features.")
    parser.add_argument(
        "--first3-labeled-features",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "synthetic_review_level_tests"
        / "emotion_rich_v2_3reviews"
        / "supervised_eval_gemini_browser"
        / "review_level_features_labeled_g25pro.csv",
    )
    parser.add_argument(
        "--reviews4x6-features",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "synthetic_review_level_tests"
        / "emotion_rich_v2_reviews4x6"
        / "features"
        / "review_level_features_unlabeled.csv",
    )
    parser.add_argument(
        "--reviews4x6-labels",
        type=Path,
        default=PROJECT_ROOT
        / "labels"
        / "gemini_browser_review_level_reviews4x6"
        / "gemini_browser_labels_emotion_rich_v2_reviews4x6_v1_pipeline.csv",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "synthetic_review_level_tests"
        / "emotion_rich_v2_6reviews"
        / "supervised_eval_gemini_browser",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    first3 = pd.read_csv(args.first3_labeled_features)
    reviews4x6 = load_labeled_feature_frame(args.reviews4x6_features, args.reviews4x6_labels)

    common_columns = [column for column in first3.columns if column in reviews4x6.columns]
    missing_from_reviews4x6 = sorted(set(first3.columns) - set(reviews4x6.columns))
    missing_from_first3 = sorted(set(reviews4x6.columns) - set(first3.columns))
    if missing_from_reviews4x6 or missing_from_first3:
        raise ValueError(
            "Input schemas differ. "
            f"Missing from reviews4x6: {missing_from_reviews4x6}; "
            f"missing from first3: {missing_from_first3}"
        )

    combined = pd.concat([first3[common_columns], reviews4x6[common_columns]], ignore_index=True)
    perfume_order = {perfume_id: index for index, perfume_id in enumerate(first3["perfume_id"].astype(str).drop_duplicates())}
    combined["_perfume_order"] = combined["perfume_id"].astype(str).map(perfume_order)
    if combined["_perfume_order"].isna().any():
        missing = sorted(combined.loc[combined["_perfume_order"].isna(), "perfume_id"].astype(str).unique().tolist())[:10]
        raise ValueError(f"Reviews4x6 contains perfume IDs not present in first3 file: {missing}")

    combined["review_index"] = pd.to_numeric(combined["review_index"], errors="raise").astype(int)
    combined = (
        combined.sort_values(["_perfume_order", "review_index", "synthetic_review_id"])
        .drop(columns=["_perfume_order"])
        .reset_index(drop=True)
    )
    validation = validate_combined(combined)

    labeled_path = args.out_root / "review_level_features_labeled_g25pro.csv"
    report_path = args.out_root / "review_level_features_labeled_g25pro_report.json"
    combined.to_csv(labeled_path, index=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "first3_labeled_features": stringify_path(args.first3_labeled_features),
            "reviews4x6_features": stringify_path(args.reviews4x6_features),
            "reviews4x6_labels": stringify_path(args.reviews4x6_labels),
        },
        "outputs": {
            "combined_labeled_features": stringify_path(labeled_path),
            "report_json": stringify_path(report_path),
        },
        "validation": validation,
        "note": "Combined emotion-rich v2 synthetic review-level features for reviews 1 through 6.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote combined labeled features: {labeled_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
