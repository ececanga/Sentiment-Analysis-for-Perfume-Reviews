# Date Created: 2026-06-12
# Date Last Modified: 2026-06-12

"""Compare emotion-rich synthetic pilot features with the original 784 benchmark.

The synthetic v2 pilot is intentionally unlabeled, so this script does not train or
score a supervised model on it. It compares the same feature families used by the
logistic regression pipeline and records the existing OG logistic regression
benchmark for context.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.labeling.strict_schema import EMOTIONS, VAD_DIMS, normalize_perfume_id
from src.synthetic.build_unlabeled_review_level_features import NRC_EMOTIONS, load_nrc_lexicon
from src.training.run_logistic_regression_benchmark import MODEL_SPEC as LOGISTIC_MODEL_SPEC
from src.utils.paths import PROJECT_ROOT, data_path

FEATURE_PREFIXES = ("NRC_", "VAD_", "Hartmann_", "Savani_", "GoEmotions_", "Cardiff_")
LABEL_COLUMNS = set(EMOTIONS + VAD_DIMS)
TRANSFORMER_LABEL_COLUMNS = ["JHartmann_Label", "Savani_Label", "GoEmotions_Label", "Cardiff_Label"]


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def feature_family(column: str) -> str:
    for prefix in FEATURE_PREFIXES:
        if column.startswith(prefix):
            return prefix.rstrip("_")
    return "other"


def shared_numeric_feature_columns(synthetic_df: pd.DataFrame, og_df: pd.DataFrame) -> list[str]:
    shared = []
    for column in synthetic_df.columns:
        if not column.startswith(FEATURE_PREFIXES) or column not in og_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(synthetic_df[column]) and pd.api.types.is_numeric_dtype(og_df[column]):
            shared.append(column)
    return shared


def aggregate_synthetic_to_perfume_level(synthetic_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    grouped = synthetic_df.groupby("perfume_id", as_index=False)[feature_cols].mean()
    counts = (
        synthetic_df.groupby("perfume_id", as_index=False)
        .agg(Perfume=("Perfume", "first"), synthetic_review_count=("synthetic_review_id", "count"))
    )
    return counts.merge(grouped, on="perfume_id", how="inner")


def build_feature_delta_table(
    synthetic_perfume_df: pd.DataFrame,
    og_same_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    merged = synthetic_perfume_df[["perfume_id", "Perfume", *feature_cols]].merge(
        og_same_df[["perfume_id", *feature_cols]],
        on="perfume_id",
        how="inner",
        suffixes=("_synthetic_mean", "_og"),
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        for feature in feature_cols:
            synthetic_value = float(row[f"{feature}_synthetic_mean"])
            og_value = float(row[f"{feature}_og"])
            delta = synthetic_value - og_value
            rows.append(
                {
                    "perfume_id": row["perfume_id"],
                    "Perfume": row["Perfume"],
                    "feature": feature,
                    "feature_family": feature_family(feature),
                    "synthetic_mean": synthetic_value,
                    "og_value": og_value,
                    "delta_synthetic_minus_og": delta,
                    "abs_delta": abs(delta),
                }
            )
    return pd.DataFrame(rows)


def summarize_feature_deltas(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    return (
        delta_df.groupby("feature_family", as_index=False)
        .agg(
            feature_count=("feature", "nunique"),
            compared_values=("feature", "size"),
            synthetic_mean=("synthetic_mean", "mean"),
            og_mean=("og_value", "mean"),
            mean_delta_synthetic_minus_og=("delta_synthetic_minus_og", "mean"),
            mean_abs_delta=("abs_delta", "mean"),
            max_abs_delta=("abs_delta", "max"),
        )
        .sort_values("feature_family")
    )


def summarize_feature_columns(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    return (
        delta_df.groupby(["feature_family", "feature"], as_index=False)
        .agg(
            synthetic_mean=("synthetic_mean", "mean"),
            og_mean=("og_value", "mean"),
            mean_delta_synthetic_minus_og=("delta_synthetic_minus_og", "mean"),
            mean_abs_delta=("abs_delta", "mean"),
        )
        .sort_values("mean_abs_delta", ascending=False)
    )


def build_transformer_label_distribution(synthetic_df: pd.DataFrame, og_same_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_name, frame in [("synthetic_v2_review_level", synthetic_df), ("og_same20_perfume_level", og_same_df)]:
        for column in TRANSFORMER_LABEL_COLUMNS:
            if column not in frame.columns:
                continue
            counts = frame[column].fillna("").astype(str).value_counts(dropna=False)
            total = int(counts.sum())
            for label, count in counts.items():
                rows.append(
                    {
                        "source": source_name,
                        "label_column": column,
                        "label": label,
                        "count": int(count),
                        "share": float(count / total) if total else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def nrc_rate_frame(review_df: pd.DataFrame, *, id_col: str, text_col: str) -> pd.DataFrame:
    lexicon = load_nrc_lexicon(data_path("lexicons", "nrc_emotion", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"))
    rows: list[dict[str, Any]] = []
    for _, source_row in review_df.iterrows():
        tokens = re.findall(r"\b\w+\b", str(source_row[text_col]).lower())
        out: dict[str, Any] = {"perfume_id": source_row[id_col], "word_count": len(tokens)}
        for emotion in NRC_EMOTIONS:
            out[f"NRC_{emotion}"] = 0
        for token in tokens:
            for emotion in lexicon.get(token, set()):
                if emotion in NRC_EMOTIONS:
                    out[f"NRC_{emotion}"] += 1
        rows.append(out)

    count_df = pd.DataFrame(rows)
    if count_df.empty:
        return pd.DataFrame()
    grouped = count_df.groupby("perfume_id", as_index=False)[["word_count", *[f"NRC_{e}" for e in NRC_EMOTIONS]]].sum()
    for emotion in NRC_EMOTIONS:
        count_col = f"NRC_{emotion}"
        grouped[f"{count_col}_per_100_words"] = grouped[count_col] / grouped["word_count"].clip(lower=1) * 100.0
    return grouped


def build_nrc_per_100_word_comparison(
    synthetic_df: pd.DataFrame,
    og_raw_reviews_path: Path,
    shared_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not og_raw_reviews_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    og_raw_df = pd.read_csv(og_raw_reviews_path)
    if "Perfume" not in og_raw_df.columns or "Final Transcript" not in og_raw_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    synthetic_text_df = synthetic_df[["perfume_id", "Final Transcript"]].copy()
    og_raw_df = og_raw_df[["Perfume", "Final Transcript"]].copy()
    og_raw_df["perfume_id"] = og_raw_df["Perfume"].map(normalize_perfume_id)
    og_raw_same_df = og_raw_df.loc[og_raw_df["perfume_id"].astype(str).isin(shared_ids)].copy()

    synthetic_rates = nrc_rate_frame(synthetic_text_df, id_col="perfume_id", text_col="Final Transcript")
    og_rates = nrc_rate_frame(og_raw_same_df, id_col="perfume_id", text_col="Final Transcript")
    if synthetic_rates.empty or og_rates.empty:
        return pd.DataFrame(), pd.DataFrame()

    rate_cols = [f"NRC_{emotion}_per_100_words" for emotion in NRC_EMOTIONS]
    merged = synthetic_rates[["perfume_id", "word_count", *rate_cols]].merge(
        og_rates[["perfume_id", "word_count", *rate_cols]],
        on="perfume_id",
        how="inner",
        suffixes=("_synthetic", "_og"),
    )

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        for emotion in NRC_EMOTIONS:
            feature = f"NRC_{emotion}"
            syn = float(row[f"{feature}_per_100_words_synthetic"])
            og = float(row[f"{feature}_per_100_words_og"])
            rows.append(
                {
                    "perfume_id": row["perfume_id"],
                    "feature": feature,
                    "synthetic_word_count": int(row["word_count_synthetic"]),
                    "og_word_count": int(row["word_count_og"]),
                    "synthetic_per_100_words": syn,
                    "og_per_100_words": og,
                    "delta_synthetic_minus_og": syn - og,
                    "abs_delta": abs(syn - og),
                }
            )
    by_perfume = pd.DataFrame(rows)
    summary = (
        by_perfume.groupby("feature", as_index=False)
        .agg(
            synthetic_per_100_words=("synthetic_per_100_words", "mean"),
            og_per_100_words=("og_per_100_words", "mean"),
            mean_delta_synthetic_minus_og=("delta_synthetic_minus_og", "mean"),
            mean_abs_delta=("abs_delta", "mean"),
        )
        .sort_values("mean_abs_delta", ascending=False)
    )
    return by_perfume, summary


def predict_synthetic_with_og_logistic(
    *,
    og_df: pd.DataFrame,
    synthetic_perfume_df: pd.DataFrame,
    metrics_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not metrics_path.exists():
        return pd.DataFrame(), pd.DataFrame(), {"status": "missing_metrics", "path": stringify_path(metrics_path)}

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    feature_cols = [col for col in metrics.get("feature_columns", []) if col in og_df.columns and col in synthetic_perfume_df.columns]
    target_cols = [col for col in metrics.get("target_columns", []) if col in og_df.columns]
    threshold = float(metrics.get("threshold", 0.3))
    model_seed = int(metrics.get("model_seed", 42))

    if not feature_cols or not target_cols:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "status": "unavailable",
                "reason": "Missing shared feature columns or target columns.",
                "shared_feature_count": len(feature_cols),
                "target_count": len(target_cols),
            },
        )

    train_df = og_df.dropna(subset=target_cols).copy()
    X_train = train_df[feature_cols]
    X_synthetic = synthetic_perfume_df[feature_cols]
    prediction_df = synthetic_perfume_df[["perfume_id", "Perfume", "synthetic_review_count"]].copy()

    summary_rows: list[dict[str, Any]] = []
    for target in target_cols:
        y_train = pd.to_numeric(train_df[target], errors="coerce").fillna(0).astype(int)
        if y_train.nunique() < 2:
            continue
        model = LOGISTIC_MODEL_SPEC.build_model(model_seed, y_train.to_numpy())
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_synthetic)
        class_list = list(model.named_steps["clf"].classes_) if hasattr(model, "named_steps") else list(model.classes_)
        positive_idx = class_list.index(1) if 1 in class_list else len(class_list) - 1
        probs = proba[:, positive_idx]
        preds = (probs >= threshold).astype(int)
        prediction_df[f"prob_{target}"] = probs
        prediction_df[f"pred_{target}"] = preds
        summary_rows.append(
            {
                "label": target,
                "threshold": threshold,
                "og_true_positive_rate": float(y_train.mean()),
                "synthetic_predicted_positive_rate": float(preds.mean()),
                "synthetic_mean_probability": float(probs.mean()),
                "synthetic_min_probability": float(probs.min()),
                "synthetic_max_probability": float(probs.max()),
                "synthetic_positive_count": int(preds.sum()),
                "synthetic_rows": int(len(preds)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    status = {
        "status": "available_inference_only",
        "training_rows": int(len(train_df)),
        "synthetic_rows": int(len(synthetic_perfume_df)),
        "feature_count": int(len(feature_cols)),
        "target_columns": target_cols,
        "threshold": threshold,
        "note": "Predictions are generated by fitting LR on OG labels and applying it to synthetic v2 features. They are not synthetic ground-truth labels and do not produce accuracy metrics.",
    }
    return prediction_df, summary_df, status


def load_og_logistic_summary(metrics_path: Path, per_label_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {"status": "missing", "path": stringify_path(metrics_path)}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_label = pd.read_csv(per_label_path) if per_label_path.exists() else pd.DataFrame()
    return {
        "status": "available",
        "path": stringify_path(metrics_path),
        "n_rows": int(metrics.get("n_rows", 0)),
        "threshold": metrics.get("threshold"),
        "target_columns": metrics.get("target_columns", []),
        "excluded_target_columns": metrics.get("excluded_target_columns", {}),
        "macro_f1": metrics.get("macro_f1"),
        "macro_roc_auc": metrics.get("macro_roc_auc"),
        "macro_pr_auc": metrics.get("macro_pr_auc"),
        "micro_f1": metrics.get("secondary_metrics", {}).get("micro_f1"),
        "subset_accuracy": metrics.get("secondary_metrics", {}).get("subset_accuracy"),
        "hamming_loss": metrics.get("secondary_metrics", {}).get("hamming_loss"),
        "per_label_rows": int(len(per_label)),
    }


def dataframe_to_markdown(frame: pd.DataFrame, *, max_rows: int | None = None, floatfmt: str = ".4f") -> str:
    out = frame.head(max_rows).copy() if max_rows is not None else frame.copy()
    if out.empty:
        return ""
    columns = list(out.columns)
    rows = []
    for _, row in out.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        rows.append(values)

    def clean_cell(value: object) -> str:
        return str(value).replace("\n", " ").replace("|", "\\|")

    header = "| " + " | ".join(clean_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(clean_cell(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_markdown_report(
    path: Path,
    *,
    report: dict[str, Any],
    family_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    label_distribution: pd.DataFrame,
    nrc_rate_summary: pd.DataFrame,
    inference_summary: pd.DataFrame,
) -> None:
    og = report["og_logistic_regression"]
    lines = [
        "# Emotion-Rich V2 Pilot vs OG Comparison",
        "",
        "## Scope",
        "",
        "- Synthetic v2 rows: "
        f"{report['synthetic_review_level']['rows']} reviews, "
        f"{report['synthetic_review_level']['unique_perfumes']} perfumes.",
        "- Synthetic v2 reviews were aggregated to perfume level before comparing with OG.",
        f"- Shared perfume count: {report['comparison_scope']['shared_perfumes']}.",
        f"- Shared numeric feature count: {report['comparison_scope']['shared_numeric_features']}.",
        "- Supervised synthetic metrics were not computed because the v2 pilot has no emotion/VAD labels.",
        "",
        "## Existing OG Logistic Regression Benchmark",
        "",
        f"- Rows: {og.get('n_rows')}",
        f"- Threshold: {og.get('threshold')}",
        f"- Macro F1: {og.get('macro_f1'):.4f}" if og.get("macro_f1") is not None else "- Macro F1: n/a",
        f"- Macro ROC AUC: {og.get('macro_roc_auc'):.4f}" if og.get("macro_roc_auc") is not None else "- Macro ROC AUC: n/a",
        f"- Macro PR AUC: {og.get('macro_pr_auc'):.4f}" if og.get("macro_pr_auc") is not None else "- Macro PR AUC: n/a",
        f"- Micro F1: {og.get('micro_f1'):.4f}" if og.get("micro_f1") is not None else "- Micro F1: n/a",
        "",
        "## Synthetic V2 LR Inference",
        "",
    ]
    if inference_summary.empty:
        lines.append("No synthetic inference output was available.")
    else:
        lines.append(dataframe_to_markdown(inference_summary, floatfmt=".4f"))
        lines.append("")
        lines.append(
            "These are classifier predictions after training on OG labels. They are useful for a pipeline sanity check, "
            "but they are not evaluation scores and not ground-truth labels."
        )

    lines.extend(
        [
            "",
        "## Feature Family Differences",
        "",
        ]
    )
    if family_summary.empty:
        lines.append("No shared feature family comparison was available.")
    else:
        lines.append(dataframe_to_markdown(family_summary, floatfmt=".4f"))

    lines.extend(["", "## Largest Mean Feature Differences", ""])
    if feature_summary.empty:
        lines.append("No feature-level comparison was available.")
    else:
        lines.append(dataframe_to_markdown(feature_summary, max_rows=15, floatfmt=".4f"))

    lines.extend(["", "## NRC Per 100 Words", ""])
    if nrc_rate_summary.empty:
        lines.append("No normalized NRC comparison was available.")
    else:
        lines.append(dataframe_to_markdown(nrc_rate_summary, floatfmt=".4f"))
        lines.append("")
        lines.append(
            "These normalized values compare lexical emotion density. They are more readable than raw NRC counts, "
            "because OG perfume rows aggregate much longer YouTube transcripts."
        )

    lines.extend(["", "## Transformer Top Labels", ""])
    if label_distribution.empty:
        lines.append("No transformer top-label comparison was available.")
    else:
        lines.append(dataframe_to_markdown(label_distribution, floatfmt=".4f"))

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This comparison tells us whether the synthetic reviews activate the same feature-producing pipeline as OG reviews. "
            "It is not a model-accuracy comparison yet. To compare F1/ROC/PR on synthetic reviews, the pilot needs review-level labels first.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare emotion-rich synthetic v2 pilot features to OG benchmark features.")
    parser.add_argument(
        "--synthetic-features",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "synthetic_review_level_tests"
        / "emotion_rich_v2_pilot100"
        / "features"
        / "review_level_features_unlabeled.csv",
    )
    parser.add_argument(
        "--og-features",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "gemini_judge"
        / "judge_g25pro_v1"
        / "analysis_out"
        / "gemini_unified_merged.csv",
    )
    parser.add_argument(
        "--og-logistic-metrics",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "gemini_judge"
        / "judge_g25pro_v1"
        / "model_benchmark"
        / "metrics"
        / "logistic_regression.json",
    )
    parser.add_argument(
        "--og-logistic-per-label",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "gemini_judge"
        / "judge_g25pro_v1"
        / "model_benchmark"
        / "metrics"
        / "logistic_regression_per_label.csv",
    )
    parser.add_argument(
        "--og-raw-reviews",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "YouTube_Perfume_Reviews_FullyFormatted.csv",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "synthetic_review_level_tests"
        / "emotion_rich_v2_pilot100"
        / "comparison_to_og",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    synthetic_df = pd.read_csv(args.synthetic_features)
    og_df = pd.read_csv(args.og_features)

    if "perfume_id" not in synthetic_df.columns or "perfume_id" not in og_df.columns:
        raise ValueError("Both synthetic and OG feature files must contain perfume_id.")

    label_columns_found = sorted(LABEL_COLUMNS & set(synthetic_df.columns))
    feature_cols = shared_numeric_feature_columns(synthetic_df, og_df)
    if not feature_cols:
        raise ValueError("No shared numeric pipeline feature columns found.")

    synthetic_perfume_df = aggregate_synthetic_to_perfume_level(synthetic_df, feature_cols)
    shared_ids = sorted(set(synthetic_perfume_df["perfume_id"].astype(str)) & set(og_df["perfume_id"].astype(str)))
    og_same_df = og_df.loc[og_df["perfume_id"].astype(str).isin(shared_ids)].copy()
    synthetic_same_df = synthetic_perfume_df.loc[synthetic_perfume_df["perfume_id"].astype(str).isin(shared_ids)].copy()

    delta_df = build_feature_delta_table(synthetic_same_df, og_same_df, feature_cols)
    family_summary = summarize_feature_deltas(delta_df)
    feature_summary = summarize_feature_columns(delta_df)
    label_distribution = build_transformer_label_distribution(synthetic_df, og_same_df)
    nrc_rate_by_perfume, nrc_rate_summary = build_nrc_per_100_word_comparison(
        synthetic_df,
        args.og_raw_reviews,
        shared_ids,
    )
    inference_predictions, inference_summary, inference_status = predict_synthetic_with_og_logistic(
        og_df=og_df,
        synthetic_perfume_df=synthetic_same_df,
        metrics_path=args.og_logistic_metrics,
    )
    og_logistic = load_og_logistic_summary(args.og_logistic_metrics, args.og_logistic_per_label)

    synthetic_perfume_path = args.out_root / "synthetic_v2_perfume_level_features.csv"
    og_same_path = args.out_root / "og_same20_perfume_features.csv"
    delta_path = args.out_root / "feature_deltas_long.csv"
    family_path = args.out_root / "feature_family_comparison.csv"
    feature_summary_path = args.out_root / "feature_level_comparison.csv"
    label_distribution_path = args.out_root / "transformer_top_label_distribution.csv"
    nrc_rate_by_perfume_path = args.out_root / "nrc_per_100_words_by_perfume.csv"
    nrc_rate_summary_path = args.out_root / "nrc_per_100_words_summary.csv"
    inference_predictions_path = args.out_root / "synthetic_v2_logistic_inference_predictions.csv"
    inference_summary_path = args.out_root / "synthetic_v2_logistic_inference_summary.csv"
    report_json_path = args.out_root / "comparison_report.json"
    report_md_path = args.out_root / "comparison_report.md"

    synthetic_same_df.to_csv(synthetic_perfume_path, index=False)
    og_same_df.to_csv(og_same_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    family_summary.to_csv(family_path, index=False)
    feature_summary.to_csv(feature_summary_path, index=False)
    label_distribution.to_csv(label_distribution_path, index=False)
    nrc_rate_by_perfume.to_csv(nrc_rate_by_perfume_path, index=False)
    nrc_rate_summary.to_csv(nrc_rate_summary_path, index=False)
    inference_predictions.to_csv(inference_predictions_path, index=False)
    inference_summary.to_csv(inference_summary_path, index=False)

    reviews_per_perfume = synthetic_df.groupby("perfume_id")["synthetic_review_id"].count()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "synthetic_features": stringify_path(args.synthetic_features),
            "og_features": stringify_path(args.og_features),
            "og_raw_reviews": stringify_path(args.og_raw_reviews),
            "og_logistic_metrics": stringify_path(args.og_logistic_metrics),
        },
        "synthetic_review_level": {
            "rows": int(len(synthetic_df)),
            "unique_perfumes": int(synthetic_df["perfume_id"].nunique()),
            "reviews_per_perfume_min": int(reviews_per_perfume.min()),
            "reviews_per_perfume_max": int(reviews_per_perfume.max()),
            "reviews_per_perfume_mean": float(reviews_per_perfume.mean()),
            "label_columns_found": label_columns_found,
            "is_unlabeled": not label_columns_found,
        },
        "comparison_scope": {
            "shared_perfumes": int(len(shared_ids)),
            "shared_numeric_features": int(len(feature_cols)),
            "feature_families": sorted({feature_family(col) for col in feature_cols}),
            "method": "mean of 5 synthetic reviews per perfume compared with OG perfume-level feature row",
        },
        "supervised_synthetic_evaluation": {
            "status": "not_run",
            "reason": "Synthetic v2 pilot is unlabeled by design; supervised F1/ROC/PR would require review-level labels first.",
        },
        "synthetic_logistic_inference": inference_status,
        "og_logistic_regression": og_logistic,
        "outputs": {
            "synthetic_v2_perfume_level_features": stringify_path(synthetic_perfume_path),
            "og_same20_perfume_features": stringify_path(og_same_path),
            "feature_deltas_long": stringify_path(delta_path),
            "feature_family_comparison": stringify_path(family_path),
            "feature_level_comparison": stringify_path(feature_summary_path),
            "transformer_top_label_distribution": stringify_path(label_distribution_path),
            "nrc_per_100_words_by_perfume": stringify_path(nrc_rate_by_perfume_path),
            "nrc_per_100_words_summary": stringify_path(nrc_rate_summary_path),
            "synthetic_v2_logistic_inference_predictions": stringify_path(inference_predictions_path),
            "synthetic_v2_logistic_inference_summary": stringify_path(inference_summary_path),
            "comparison_report_json": stringify_path(report_json_path),
            "comparison_report_md": stringify_path(report_md_path),
        },
    }
    report_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(
        report_md_path,
        report=report,
        family_summary=family_summary,
        feature_summary=feature_summary,
        label_distribution=label_distribution,
        nrc_rate_summary=nrc_rate_summary,
        inference_summary=inference_summary,
    )

    print(f"Wrote comparison report: {report_md_path}")
    print(f"Shared perfumes: {len(shared_ids)}")
    print(f"Shared numeric features: {len(feature_cols)}")
    print("Synthetic supervised metrics: not run because v2 pilot is unlabeled")


if __name__ == "__main__":
    main()
