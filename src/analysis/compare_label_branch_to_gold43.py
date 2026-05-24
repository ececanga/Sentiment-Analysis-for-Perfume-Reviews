# Date Created: 2026-05-20

"""Compare frozen baseline and judge label branches against the strict 43-row gold packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

from src.analysis.analyze_gemini_unified import build_ids_from_names
from src.labeling.strict_schema import EMOTIONS, VAD_DIMS
from src.utils.paths import PROJECT_ROOT


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    result = pearsonr(x, y)
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    result = spearmanr(x, y)
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=PROJECT_ROOT / "labels" / "gemini_labels_784_v1.csv",
        help="Canonical frozen baseline label CSV.",
    )
    parser.add_argument(
        "--judge-csv",
        type=Path,
        default=PROJECT_ROOT / "labels" / "gemini_labels_784_judge_g25pro_v1.csv",
        help="Judge branch label CSV.",
    )
    parser.add_argument(
        "--gold-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotation_packets" / "Labeled_Annotation_Packet_strict_v1.csv",
        help="Strict 43-row gold annotation packet.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "evaluation_config.yaml",
        help="Official frozen benchmark config used for the target label list.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gemini_judge" / "judge_g25pro_v1" / "gold_eval",
        help="Output directory for gold comparison artifacts.",
    )
    return parser.parse_args()


def load_eval_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid evaluation config: {path}")
    return payload


def require_columns(df: pd.DataFrame, columns: Iterable[str], *, path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def load_label_frame(path: Path, *, all_emotions: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Label CSV not found: {path}")

    df = pd.read_csv(path)
    require_columns(df, ["perfume_id", "perfume_name", *all_emotions, *VAD_DIMS], path=path)
    df["perfume_id"] = df["perfume_id"].fillna("").astype(str).str.strip()
    df["perfume_name"] = df["perfume_name"].fillna("").astype(str).str.strip()

    if df["perfume_id"].eq("").any():
        raise ValueError(f"Blank perfume_id detected in {path}")
    dup_ids = sorted(df.loc[df["perfume_id"].duplicated(keep=False), "perfume_id"].astype(str).unique().tolist())
    if dup_ids:
        raise ValueError(f"Duplicate perfume_id values in {path}: {dup_ids[:10]}")

    for column in all_emotions:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        invalid = df[column].isna() | ~df[column].isin([0, 1])
        if invalid.any():
            raise ValueError(f"Invalid binary values in {path} for column {column}")

    for column in VAD_DIMS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"Missing VAD values in {path} for column {column}")

    return df.sort_values("perfume_id").reset_index(drop=True)


def load_gold_frame(path: Path, *, all_emotions: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Gold CSV not found: {path}")

    df = pd.read_csv(path)
    require_columns(df, ["Perfume", *all_emotions, *VAD_DIMS], path=path)

    df["perfume_name_gold"] = df["Perfume"].fillna("").astype(str).str.strip()
    if df["perfume_name_gold"].eq("").any():
        raise ValueError(f"Blank Perfume values detected in gold CSV: {path}")

    ids = build_ids_from_names(df["perfume_name_gold"].tolist())
    df["perfume_id"] = ids["perfume_id"].fillna("").astype(str).str.strip()
    df["perfume_id_hash_gold"] = ids["perfume_id_hash"].fillna("").astype(str).str.strip()

    dup_ids = sorted(df.loc[df["perfume_id"].duplicated(keep=False), "perfume_id"].astype(str).unique().tolist())
    if dup_ids:
        raise ValueError(f"Ambiguous gold matching: duplicate derived perfume_id values {dup_ids[:10]}")

    for column in all_emotions:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        invalid = df[column].isna() | ~df[column].isin([0, 1])
        if invalid.any():
            raise ValueError(f"Invalid binary gold values in {path} for column {column}")

    for column in VAD_DIMS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"Missing gold VAD values in {path} for column {column}")

    return df.sort_values("perfume_id").reset_index(drop=True)


def build_match_report(
    *,
    gold_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    baseline_csv: Path,
    judge_csv: Path,
    gold_csv: Path,
) -> Dict[str, Any]:
    gold_ids = gold_df["perfume_id"].astype(str).tolist()
    baseline_ids = set(baseline_df["perfume_id"].astype(str).tolist())
    judge_ids = set(judge_df["perfume_id"].astype(str).tolist())

    missing_in_baseline = sorted(set(gold_ids) - baseline_ids)
    missing_in_judge = sorted(set(gold_ids) - judge_ids)
    common_ids = sorted(set(gold_ids) & baseline_ids & judge_ids)

    report = {
        "matching_strategy": "build_ids_from_names(Perfume) -> perfume_id",
        "gold_csv": stringify_path(gold_csv),
        "baseline_csv": stringify_path(baseline_csv),
        "judge_csv": stringify_path(judge_csv),
        "gold_rows": int(len(gold_df)),
        "gold_rows_unique_perfume_id": int(gold_df["perfume_id"].nunique()),
        "baseline_rows": int(len(baseline_df)),
        "judge_rows": int(len(judge_df)),
        "common_match_count": int(len(common_ids)),
        "same_matched_set": bool(set(common_ids) == set(gold_ids)),
        "missing_in_baseline": missing_in_baseline,
        "missing_in_judge": missing_in_judge,
        "ambiguous_gold_perfume_ids": [],
        "matched_perfume_ids": common_ids,
    }
    return report


def validate_match_report(report: Mapping[str, Any]) -> None:
    if int(report["gold_rows"]) == 0:
        raise ValueError("Gold packet has zero rows.")
    if int(report["common_match_count"]) != int(report["gold_rows"]):
        raise ValueError(
            "Gold packet did not match baseline and judge on the same complete set. "
            f"Missing baseline={report['missing_in_baseline']} missing judge={report['missing_in_judge']}"
        )
    if not bool(report["same_matched_set"]):
        raise ValueError("Baseline and judge do not align to the same gold match set.")


def prepare_branch_eval_frame(
    *,
    branch_name: str,
    label_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    all_emotions: Sequence[str],
) -> pd.DataFrame:
    merged = gold_df.merge(
        label_df[["perfume_id", "perfume_name", *all_emotions, *VAD_DIMS]],
        on="perfume_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_gold", f"_{branch_name}"),
    )
    if len(merged) != len(gold_df):
        raise ValueError(
            f"{branch_name} did not preserve the complete gold row set after matching. "
            f"Expected {len(gold_df)} rows, found {len(merged)}."
        )

    rename_map = {"perfume_name": f"perfume_name_{branch_name}"}
    for label in all_emotions:
        rename_map[f"{label}_gold"] = label
        rename_map[f"{label}_{branch_name}"] = f"{label}_pred"
    for dimension in VAD_DIMS:
        rename_map[f"{dimension}_gold"] = dimension
        rename_map[f"{dimension}_{branch_name}"] = f"{dimension}_pred"

    return merged.rename(columns=rename_map).sort_values("perfume_id").reset_index(drop=True)


def compute_emotion_metrics(
    *,
    merged_df: pd.DataFrame,
    official_labels: Sequence[str],
    all_emotions: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    y_true_official = merged_df.loc[:, official_labels].to_numpy(dtype=int)
    y_pred_official = merged_df.loc[:, [f"{label}_pred" for label in official_labels]].to_numpy(dtype=int)

    per_label_rows: List[Dict[str, Any]] = []
    per_label_f1: Dict[str, float] = {}
    for label in all_emotions:
        true_vals = merged_df[label].to_numpy(dtype=int)
        pred_vals = merged_df[f"{label}_pred"].to_numpy(dtype=int)
        f1_value = float(f1_score(true_vals, pred_vals, zero_division=0))
        per_label_f1[label] = f1_value
        per_label_rows.append(
            {
                "label": label,
                "in_primary_macro_f1": label in set(official_labels),
                "support_pos": int(true_vals.sum()),
                "support_neg": int(len(true_vals) - true_vals.sum()),
                "gold_prevalence": float(true_vals.mean()),
                "predicted_prevalence": float(pred_vals.mean()),
                "precision": float(precision_score(true_vals, pred_vals, zero_division=0)),
                "recall": float(recall_score(true_vals, pred_vals, zero_division=0)),
                "f1": f1_value,
            }
        )

    macro_f1_official = float(np.mean([per_label_f1[label] for label in official_labels]))
    metrics = {
        "macro_f1_official": macro_f1_official,
        "macro_f1_all_emotions": float(np.mean([per_label_f1[label] for label in all_emotions])),
        "micro_f1_official": float(f1_score(y_true_official, y_pred_official, average="micro", zero_division=0)),
        "subset_accuracy_official": float(np.mean(np.all(y_true_official == y_pred_official, axis=1))),
        "hamming_loss_official": float(hamming_loss(y_true_official, y_pred_official)),
        "gold_label_cardinality_official": float(np.mean(np.sum(y_true_official, axis=1))),
        "pred_label_cardinality_official": float(np.mean(np.sum(y_pred_official, axis=1))),
        "per_label_f1": per_label_f1,
    }
    return pd.DataFrame(per_label_rows), metrics


def compute_vad_metrics(merged_df: pd.DataFrame) -> Dict[str, Any]:
    per_dim: Dict[str, Dict[str, Any]] = {}
    avg_rows: List[Dict[str, float]] = []
    for dimension in VAD_DIMS:
        gold_vals = merged_df[dimension].to_numpy(dtype=float)
        pred_vals = merged_df[f"{dimension}_pred"].to_numpy(dtype=float)
        abs_err = np.abs(gold_vals - pred_vals)
        sq_err = np.square(gold_vals - pred_vals)
        row = {
            "n": int(len(gold_vals)),
            "mae": float(abs_err.mean()),
            "rmse": float(np.sqrt(sq_err.mean())),
            "pearson_r": safe_pearson(gold_vals, pred_vals),
            "spearman_r": safe_spearman(gold_vals, pred_vals),
        }
        per_dim[dimension] = row
        avg_rows.append(row)

    average = {
        "n": int(np.mean([row["n"] for row in avg_rows])) if avg_rows else 0,
        "mae": float(np.mean([row["mae"] for row in avg_rows])) if avg_rows else float("nan"),
        "rmse": float(np.mean([row["rmse"] for row in avg_rows])) if avg_rows else float("nan"),
        "pearson_r": float(np.nanmean([row["pearson_r"] for row in avg_rows])) if avg_rows else float("nan"),
        "spearman_r": float(np.nanmean([row["spearman_r"] for row in avg_rows])) if avg_rows else float("nan"),
    }
    return {"by_dimension": per_dim, "average": average}


def build_branch_payload(
    *,
    branch_name: str,
    label_csv: Path,
    gold_csv: Path,
    official_labels: Sequence[str],
    excluded_from_primary: Sequence[str],
    emotion_metrics: Mapping[str, Any],
    vad_metrics: Mapping[str, Any],
    matched_rows: int,
) -> Dict[str, Any]:
    return {
        "branch_name": branch_name,
        "label_csv": stringify_path(label_csv),
        "gold_csv": stringify_path(gold_csv),
        "matched_gold_rows": int(matched_rows),
        "official_target_labels": list(official_labels),
        "excluded_from_primary": list(excluded_from_primary),
        "primary_decision_metric": {
            "name": "macro_f1",
            "label_scope": "official_target_labels",
            "excluded_labels": list(excluded_from_primary),
            "value": float(emotion_metrics["macro_f1_official"]),
        },
        "emotion_metrics": dict(emotion_metrics),
        "vad_metrics": dict(vad_metrics),
    }


def build_rowwise_comparison(
    *,
    gold_df: pd.DataFrame,
    baseline_eval_df: pd.DataFrame,
    judge_eval_df: pd.DataFrame,
    all_emotions: Sequence[str],
) -> pd.DataFrame:
    baseline_cols = [
        "perfume_id",
        "perfume_name_gold",
        "perfume_name_baseline",
        *all_emotions,
        *VAD_DIMS,
        *[f"{label}_pred" for label in all_emotions],
        *[f"{dimension}_pred" for dimension in VAD_DIMS],
    ]
    baseline_rename = {
        f"{label}_pred": f"{label}_baseline" for label in all_emotions
    } | {f"{dimension}_pred": f"{dimension}_baseline" for dimension in VAD_DIMS}
    judge_cols = [
        "perfume_id",
        "perfume_name_judge",
        *[f"{label}_pred" for label in all_emotions],
        *[f"{dimension}_pred" for dimension in VAD_DIMS],
    ]
    judge_rename = {
        f"{label}_pred": f"{label}_judge" for label in all_emotions
    } | {f"{dimension}_pred": f"{dimension}_judge" for dimension in VAD_DIMS}

    merged = baseline_eval_df[baseline_cols].rename(columns=baseline_rename).merge(
        judge_eval_df[judge_cols].rename(columns=judge_rename),
        on="perfume_id",
        how="inner",
        validate="one_to_one",
    )

    ordered_cols = ["perfume_id", "perfume_name_gold", "perfume_name_baseline", "perfume_name_judge"]
    for label in all_emotions:
        ordered_cols.extend(
            [
                label,
                f"{label}_baseline",
                f"{label}_judge",
            ]
        )
        merged[f"baseline_match_{label}"] = (merged[label].astype(int) == merged[f"{label}_baseline"].astype(int)).astype(int)
        merged[f"judge_match_{label}"] = (merged[label].astype(int) == merged[f"{label}_judge"].astype(int)).astype(int)
        ordered_cols.extend([f"baseline_match_{label}", f"judge_match_{label}"])

    for dimension in VAD_DIMS:
        ordered_cols.extend(
            [
                dimension,
                f"{dimension}_baseline",
                f"{dimension}_judge",
            ]
        )
        merged[f"baseline_abs_error_{dimension}"] = (merged[dimension] - merged[f"{dimension}_baseline"]).abs()
        merged[f"judge_abs_error_{dimension}"] = (merged[dimension] - merged[f"{dimension}_judge"]).abs()
        ordered_cols.extend([f"baseline_abs_error_{dimension}", f"judge_abs_error_{dimension}"])

    return merged.loc[:, ordered_cols].sort_values("perfume_id").reset_index(drop=True)


def run_comparison(
    *,
    baseline_csv: Path,
    judge_csv: Path,
    gold_csv: Path,
    config_path: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_eval_config(config_path)
    official_labels = list(config["targets"]["official_emotion_columns"])
    excluded_from_primary = list(dict(config["targets"]["excluded_emotion_columns"]).keys())
    all_emotions = [label for label in EMOTIONS if label in set(official_labels) | set(excluded_from_primary)]

    baseline_df = load_label_frame(baseline_csv, all_emotions=all_emotions)
    judge_df = load_label_frame(judge_csv, all_emotions=all_emotions)
    gold_df = load_gold_frame(gold_csv, all_emotions=all_emotions)

    match_report = build_match_report(
        gold_df=gold_df,
        baseline_df=baseline_df,
        judge_df=judge_df,
        baseline_csv=baseline_csv,
        judge_csv=judge_csv,
        gold_csv=gold_csv,
    )
    match_report_path = out_dir / "gold_packet_match_report.json"
    match_report_path.write_text(json.dumps(match_report, indent=2), encoding="utf-8")
    validate_match_report(match_report)

    baseline_eval = prepare_branch_eval_frame(
        branch_name="baseline",
        label_df=baseline_df,
        gold_df=gold_df,
        all_emotions=all_emotions,
    )
    judge_eval = prepare_branch_eval_frame(
        branch_name="judge",
        label_df=judge_df,
        gold_df=gold_df,
        all_emotions=all_emotions,
    )

    baseline_per_label_df, baseline_emotion_metrics = compute_emotion_metrics(
        merged_df=baseline_eval,
        official_labels=official_labels,
        all_emotions=all_emotions,
    )
    judge_per_label_df, judge_emotion_metrics = compute_emotion_metrics(
        merged_df=judge_eval,
        official_labels=official_labels,
        all_emotions=all_emotions,
    )
    baseline_vad_metrics = compute_vad_metrics(baseline_eval)
    judge_vad_metrics = compute_vad_metrics(judge_eval)

    baseline_payload = build_branch_payload(
        branch_name="baseline",
        label_csv=baseline_csv,
        gold_csv=gold_csv,
        official_labels=official_labels,
        excluded_from_primary=excluded_from_primary,
        emotion_metrics=baseline_emotion_metrics,
        vad_metrics=baseline_vad_metrics,
        matched_rows=len(gold_df),
    )
    judge_payload = build_branch_payload(
        branch_name="judge",
        label_csv=judge_csv,
        gold_csv=gold_csv,
        official_labels=official_labels,
        excluded_from_primary=excluded_from_primary,
        emotion_metrics=judge_emotion_metrics,
        vad_metrics=judge_vad_metrics,
        matched_rows=len(gold_df),
    )

    baseline_metrics_path = out_dir / "baseline_vs_gold_metrics.json"
    baseline_per_label_path = out_dir / "baseline_vs_gold_per_label.csv"
    judge_metrics_path = out_dir / "judge_vs_gold_metrics.json"
    judge_per_label_path = out_dir / "judge_vs_gold_per_label.csv"
    branch_compare_path = out_dir / "baseline_vs_judge_gold_comparison.csv"

    baseline_metrics_path.write_text(json.dumps(baseline_payload, indent=2), encoding="utf-8")
    judge_metrics_path.write_text(json.dumps(judge_payload, indent=2), encoding="utf-8")
    baseline_per_label_df.to_csv(baseline_per_label_path, index=False)
    judge_per_label_df.to_csv(judge_per_label_path, index=False)

    rowwise_df = build_rowwise_comparison(
        gold_df=gold_df,
        baseline_eval_df=baseline_eval,
        judge_eval_df=judge_eval,
        all_emotions=all_emotions,
    )
    rowwise_df.to_csv(branch_compare_path, index=False)

    print(f"Matched gold rows: {len(gold_df)}")
    print(
        "Primary macro-F1 "
        f"baseline={baseline_payload['primary_decision_metric']['value']:.6f} "
        f"judge={judge_payload['primary_decision_metric']['value']:.6f}"
    )
    print(f"Wrote: {baseline_metrics_path}")
    print(f"Wrote: {baseline_per_label_path}")
    print(f"Wrote: {judge_metrics_path}")
    print(f"Wrote: {judge_per_label_path}")
    print(f"Wrote: {branch_compare_path}")
    print(f"Wrote: {match_report_path}")

    return {
        "baseline_metrics_path": baseline_metrics_path,
        "judge_metrics_path": judge_metrics_path,
        "match_report_path": match_report_path,
        "matched_gold_rows": len(gold_df),
        "baseline_primary_macro_f1": baseline_payload["primary_decision_metric"]["value"],
        "judge_primary_macro_f1": judge_payload["primary_decision_metric"]["value"],
    }


def main() -> None:
    args = parse_args()
    run_comparison(
        baseline_csv=args.baseline_csv,
        judge_csv=args.judge_csv,
        gold_csv=args.gold_csv,
        config_path=args.config,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
