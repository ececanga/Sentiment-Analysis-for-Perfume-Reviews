"""Leakage-safe threshold tuning utilities for multi-label benchmarks.

In imbalanced multi-label classification, a fixed baseline threshold may be
suboptimal because label prevalences and score distributions differ across
labels. Threshold tuning is used here as a calibration step to improve final
binary decisions under Macro-F1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, roc_auc_score
from sklearn.model_selection import KFold

THRESHOLD_TUNING_RATIONALE = (
    "In imbalanced multi-label classification, a fixed baseline threshold may be suboptimal "
    "because label prevalences and score distributions differ across labels. Threshold tuning "
    "is used here as a calibration step to improve final binary decisions under Macro-F1."
)
SUPPORTED_THRESHOLD_STRATEGIES = ("global", "per_label")


@dataclass(frozen=True)
class ThresholdTuningSettings:
    enabled: bool
    strategies: Tuple[str, ...]
    grid: Tuple[float, ...]
    selection_metric: str
    baseline_threshold: float
    tie_break: str


def _metric_or_nan(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def _mean_finite(values: Sequence[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def _round_threshold(value: float) -> float:
    return float(round(float(value), 10))


def build_threshold_grid(*, start: float, end: float, step: float) -> Tuple[float, ...]:
    if step <= 0.0:
        raise ValueError("Threshold grid step must be positive.")
    if start < 0.0 or end > 1.0 or start > end:
        raise ValueError("Threshold grid must satisfy 0.0 <= start <= end <= 1.0.")

    values: List[float] = []
    current = float(start)
    epsilon = step / 10_000.0
    while current <= end + epsilon:
        values.append(_round_threshold(current))
        current += step
    return tuple(dict.fromkeys(values))


def load_threshold_tuning_settings(
    config: Mapping[str, Any],
    *,
    baseline_threshold: float,
) -> ThresholdTuningSettings:
    evaluation_cfg = config.get("evaluation", {})
    tuning_cfg = evaluation_cfg.get("threshold_tuning", {}) if isinstance(evaluation_cfg, Mapping) else {}
    if not isinstance(tuning_cfg, Mapping) or not bool(tuning_cfg.get("enabled", False)):
        return ThresholdTuningSettings(
            enabled=False,
            strategies=(),
            grid=(),
            selection_metric="macro_f1",
            baseline_threshold=float(baseline_threshold),
            tie_break="closest_to_baseline",
        )

    strategy_values = tuning_cfg.get("strategies", SUPPORTED_THRESHOLD_STRATEGIES)
    if isinstance(strategy_values, str):
        strategies = (strategy_values,)
    else:
        strategies = tuple(str(value) for value in strategy_values)
    if not strategies:
        raise ValueError("At least one threshold tuning strategy must be configured.")
    invalid_strategies = [strategy for strategy in strategies if strategy not in SUPPORTED_THRESHOLD_STRATEGIES]
    if invalid_strategies:
        raise ValueError(f"Unsupported threshold tuning strategies: {invalid_strategies}")

    selection_metric = str(tuning_cfg.get("selection_metric", "macro_f1"))
    if selection_metric != "macro_f1":
        raise ValueError("Threshold tuning currently supports selection_metric='macro_f1' only.")

    grid_cfg = tuning_cfg.get("threshold_grid", {})
    if not isinstance(grid_cfg, Mapping):
        raise ValueError("evaluation.threshold_tuning.threshold_grid must be a mapping.")

    if "values" in grid_cfg:
        raw_values = grid_cfg["values"]
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
            raise ValueError("threshold_grid.values must be a sequence of numeric thresholds.")
        grid = tuple(sorted({_round_threshold(float(value)) for value in raw_values}))
    else:
        grid = build_threshold_grid(
            start=float(grid_cfg.get("start", 0.05)),
            end=float(grid_cfg.get("end", 0.95)),
            step=float(grid_cfg.get("step", 0.05)),
        )

    if not grid:
        raise ValueError("Threshold tuning grid must contain at least one threshold.")
    if any(value < 0.0 or value > 1.0 for value in grid):
        raise ValueError("Threshold tuning grid values must be between 0.0 and 1.0.")

    tie_break = str(tuning_cfg.get("tie_break", "closest_to_baseline"))
    if tie_break != "closest_to_baseline":
        raise ValueError("Threshold tuning currently supports tie_break='closest_to_baseline' only.")

    return ThresholdTuningSettings(
        enabled=True,
        strategies=strategies,
        grid=grid,
        selection_metric=selection_metric,
        baseline_threshold=float(baseline_threshold),
        tie_break=tie_break,
    )


def build_inner_validation_manifest(
    df: pd.DataFrame,
    *,
    unit_id_col: str,
    fold_id_col: str,
    display_name_col: str | None,
    inner_n_splits: int,
    seed: int,
) -> pd.DataFrame:
    if inner_n_splits < 2:
        raise ValueError("Inner validation n_splits must be >= 2.")

    columns = ["outer_fold_id", unit_id_col]
    if display_name_col is not None:
        columns.append(display_name_col)
    columns.append("inner_role")

    rows: List[Dict[str, Any]] = []
    for outer_fold_id in sorted(pd.unique(df[fold_id_col]).tolist()):
        outer_train_df = df.loc[df[fold_id_col] != outer_fold_id].sort_values(unit_id_col).reset_index(drop=True)
        n_splits = min(inner_n_splits, len(outer_train_df))
        if n_splits < 2:
            raise ValueError(f"Outer fold {outer_fold_id} has too few training rows for inner validation.")

        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        _, val_positions = next(splitter.split(outer_train_df))
        val_ids = set(outer_train_df.iloc[val_positions][unit_id_col].astype(str).tolist())

        for _, row in outer_train_df.iterrows():
            row_payload: Dict[str, Any] = {
                "outer_fold_id": int(outer_fold_id),
                unit_id_col: str(row[unit_id_col]),
                "inner_role": "val" if str(row[unit_id_col]) in val_ids else "train",
            }
            if display_name_col is not None:
                row_payload[display_name_col] = str(row.get(display_name_col, ""))
            rows.append(row_payload)

    return pd.DataFrame(rows, columns=columns)


def load_or_write_inner_validation_manifest(
    expected_df: pd.DataFrame,
    *,
    output_path: Path,
) -> pd.DataFrame:
    sort_cols = ["outer_fold_id", expected_df.columns[1]]
    expected_sorted = expected_df.sort_values(sort_cols).reset_index(drop=True)
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        if list(existing_df.columns) != list(expected_df.columns):
            raise ValueError(f"Existing inner validation manifest has unexpected columns: {output_path}")
        existing_sorted = existing_df.sort_values(sort_cols).reset_index(drop=True)
        if not expected_sorted.equals(existing_sorted):
            raise ValueError(f"Existing inner validation manifest does not match the frozen rule: {output_path}")
        return existing_sorted

    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_sorted.to_csv(output_path, index=False)
    return expected_sorted


def _validate_probability_inputs(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_cols: Sequence[str],
) -> None:
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape.")
    if y_true.ndim != 2:
        raise ValueError("y_true and y_prob must be two-dimensional arrays.")
    if y_true.shape[1] != len(target_cols):
        raise ValueError("target_cols length must match the probability matrix label dimension.")


def _threshold_tie_key(threshold: float, baseline_threshold: float) -> Tuple[float, float]:
    return (abs(float(threshold) - float(baseline_threshold)), float(threshold))


def _per_label_f1_values(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    return [
        _metric_or_nan(f1_score, y_true[:, label_idx], y_pred[:, label_idx], zero_division=0)
        for label_idx in range(y_true.shape[1])
    ]


def _select_single_threshold(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: Sequence[float],
    baseline_threshold: float,
) -> Tuple[float, float, List[float]]:
    best_threshold = float(grid[0])
    best_score = -1.0
    best_per_label_f1: List[float] = []

    for threshold in grid:
        y_pred = (y_prob >= threshold).astype(int)
        per_label_f1 = _per_label_f1_values(y_true, y_pred)
        score = float(np.mean(per_label_f1))
        is_better = score > best_score + 1e-12
        is_tie = abs(score - best_score) <= 1e-12
        if is_better or (
            is_tie
            and _threshold_tie_key(threshold, baseline_threshold)
            < _threshold_tie_key(best_threshold, baseline_threshold)
        ):
            best_threshold = float(threshold)
            best_score = score
            best_per_label_f1 = per_label_f1

    return best_threshold, best_score, best_per_label_f1


def _select_per_label_thresholds(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_cols: Sequence[str],
    grid: Sequence[float],
    baseline_threshold: float,
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    thresholds: Dict[str, float] = {}
    validation_per_label_f1: Dict[str, float] = {}

    for label_idx, label in enumerate(target_cols):
        best_threshold = float(grid[0])
        best_score = -1.0
        for threshold in grid:
            y_pred = (y_prob[:, label_idx] >= threshold).astype(int)
            score = _metric_or_nan(f1_score, y_true[:, label_idx], y_pred, zero_division=0)
            is_better = score > best_score + 1e-12
            is_tie = abs(score - best_score) <= 1e-12
            if is_better or (
                is_tie
                and _threshold_tie_key(threshold, baseline_threshold)
                < _threshold_tie_key(best_threshold, baseline_threshold)
            ):
                best_threshold = float(threshold)
                best_score = float(score)

        thresholds[label] = best_threshold
        validation_per_label_f1[label] = best_score

    return thresholds, float(np.mean(list(validation_per_label_f1.values()))), validation_per_label_f1


def select_fold_thresholds(
    *,
    y_val: np.ndarray,
    prob_val: np.ndarray,
    target_cols: Sequence[str],
    fold_id: int,
    strategy: str,
    settings: ThresholdTuningSettings,
) -> Dict[str, Any]:
    _validate_probability_inputs(y_true=y_val, y_prob=prob_val, target_cols=target_cols)
    if strategy not in settings.strategies:
        raise ValueError(f"Strategy {strategy!r} is not enabled in threshold tuning settings.")

    if strategy == "global":
        threshold, validation_macro_f1, per_label_f1 = _select_single_threshold(
            y_true=y_val,
            y_prob=prob_val,
            grid=settings.grid,
            baseline_threshold=settings.baseline_threshold,
        )
        thresholds = {label: float(threshold) for label in target_cols}
        validation_per_label_f1 = {
            label: float(per_label_f1[label_idx]) for label_idx, label in enumerate(target_cols)
        }
    elif strategy == "per_label":
        thresholds, validation_macro_f1, validation_per_label_f1 = _select_per_label_thresholds(
            y_true=y_val,
            y_prob=prob_val,
            target_cols=target_cols,
            grid=settings.grid,
            baseline_threshold=settings.baseline_threshold,
        )
    else:
        raise ValueError(f"Unsupported threshold tuning strategy: {strategy}")

    return {
        "fold_id": int(fold_id),
        "strategy": strategy,
        "thresholds": thresholds,
        "validation_macro_f1": float(validation_macro_f1),
        "validation_per_label_f1": validation_per_label_f1,
        "n_validation_rows": int(len(y_val)),
    }


def compute_metrics_from_predictions(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    target_cols: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    _validate_probability_inputs(y_true=y_true, y_prob=y_prob, target_cols=target_cols)
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape.")

    per_label_rows: List[Dict[str, Any]] = []
    roc_values: List[float] = []
    pr_values: List[float] = []

    for label_idx, label in enumerate(target_cols):
        yt = y_true[:, label_idx]
        yp = y_pred[:, label_idx]
        yp_prob = y_prob[:, label_idx]
        roc_auc = _metric_or_nan(roc_auc_score, yt, yp_prob)
        pr_auc = _metric_or_nan(average_precision_score, yt, yp_prob)
        roc_values.append(roc_auc)
        pr_values.append(pr_auc)
        per_label_rows.append(
            {
                "label": label,
                "support_pos": int(yt.sum()),
                "support_neg": int(len(yt) - yt.sum()),
                "f1": _metric_or_nan(f1_score, yt, yp, zero_division=0),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

    metrics = {
        "macro_f1": float(np.mean([row["f1"] for row in per_label_rows])),
        "macro_roc_auc": _mean_finite(roc_values),
        "macro_pr_auc": _mean_finite(pr_values),
        "micro_f1": _metric_or_nan(f1_score, y_true, y_pred, average="micro", zero_division=0),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": _metric_or_nan(hamming_loss, y_true, y_pred),
        "label_cardinality": float(np.mean(np.sum(y_true, axis=1))),
    }
    return per_label_rows, metrics


def apply_fold_thresholds(
    *,
    y_prob: np.ndarray,
    fold_ids: np.ndarray,
    target_cols: Sequence[str],
    selections: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    y_pred = np.zeros(y_prob.shape, dtype=int)
    selections_by_fold = {int(selection["fold_id"]): selection for selection in selections}

    for fold_id in sorted(pd.unique(fold_ids).tolist()):
        if int(fold_id) not in selections_by_fold:
            raise ValueError(f"Missing threshold selection for fold {fold_id}.")
        selection = selections_by_fold[int(fold_id)]
        threshold_vector = np.asarray(
            [float(selection["thresholds"][label]) for label in target_cols],
            dtype=float,
        )
        fold_mask = fold_ids == fold_id
        y_pred[fold_mask] = (y_prob[fold_mask] >= threshold_vector).astype(int)

    return y_pred


def summarize_threshold_tuning_strategy(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fold_ids: np.ndarray,
    target_cols: Sequence[str],
    selections: Sequence[Mapping[str, Any]],
    baseline_per_label_rows: Sequence[Mapping[str, Any]],
    baseline_metrics: Mapping[str, float],
) -> Tuple[Dict[str, Any], np.ndarray, List[Dict[str, Any]], Dict[str, float]]:
    if not selections:
        raise ValueError("At least one fold threshold selection is required.")

    strategy = str(selections[0]["strategy"])
    y_pred = apply_fold_thresholds(
        y_prob=y_prob,
        fold_ids=fold_ids,
        target_cols=target_cols,
        selections=selections,
    )
    tuned_per_label_rows, tuned_metrics = compute_metrics_from_predictions(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        target_cols=target_cols,
    )

    baseline_f1_by_label = {
        str(row["label"]): float(row["f1"]) for row in baseline_per_label_rows
    }
    tuned_f1_by_label = {
        str(row["label"]): float(row["f1"]) for row in tuned_per_label_rows
    }
    per_label_f1_report = []
    for label in target_cols:
        thresholds_by_fold = {
            str(selection["fold_id"]): float(selection["thresholds"][label])
            for selection in selections
        }
        baseline_f1 = float(baseline_f1_by_label[label])
        tuned_f1 = float(tuned_f1_by_label[label])
        per_label_f1_report.append(
            {
                "label": label,
                "baseline_f1": baseline_f1,
                "tuned_f1": tuned_f1,
                "delta_f1": tuned_f1 - baseline_f1,
                "selected_thresholds_by_fold": thresholds_by_fold,
            }
        )

    fold_report = []
    for selection in selections:
        if strategy == "global":
            fold_thresholds: Dict[str, Any] = {
                "threshold": float(next(iter(selection["thresholds"].values())))
            }
        else:
            fold_thresholds = {
                "thresholds": {
                    label: float(selection["thresholds"][label])
                    for label in target_cols
                }
            }

        fold_report.append(
            {
                "fold_id": int(selection["fold_id"]),
                **fold_thresholds,
                "validation_macro_f1": float(selection["validation_macro_f1"]),
                "validation_per_label_f1": {
                    label: float(selection["validation_per_label_f1"][label])
                    for label in target_cols
                },
                "n_validation_rows": int(selection["n_validation_rows"]),
            }
        )

    report = {
        "strategy": strategy,
        "baseline_macro_f1": float(baseline_metrics["macro_f1"]),
        "tuned_macro_f1": float(tuned_metrics["macro_f1"]),
        "macro_f1_delta": float(tuned_metrics["macro_f1"] - baseline_metrics["macro_f1"]),
        "secondary_metrics": {
            "micro_f1": tuned_metrics["micro_f1"],
            "subset_accuracy": tuned_metrics["subset_accuracy"],
            "hamming_loss": tuned_metrics["hamming_loss"],
            "label_cardinality": tuned_metrics["label_cardinality"],
        },
        "selected_thresholds_by_fold": fold_report,
        "per_label_f1": per_label_f1_report,
    }
    return report, y_pred, tuned_per_label_rows, tuned_metrics


def build_threshold_tuning_report(
    *,
    settings: ThresholdTuningSettings,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fold_ids: np.ndarray,
    target_cols: Sequence[str],
    selections_by_strategy: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline_per_label_rows: Sequence[Mapping[str, Any]],
    baseline_metrics: Mapping[str, float],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    if not settings.enabled:
        return {"enabled": False}, {}

    report: Dict[str, Any] = {
        "enabled": True,
        "rationale": THRESHOLD_TUNING_RATIONALE,
        "baseline_threshold": float(settings.baseline_threshold),
        "selection_metric": settings.selection_metric,
        "grid": [float(value) for value in settings.grid],
        "tie_break": settings.tie_break,
        "baseline_macro_f1": float(baseline_metrics["macro_f1"]),
        "strategies": {},
    }
    tuned_predictions: Dict[str, np.ndarray] = {}

    for strategy in settings.strategies:
        strategy_report, y_pred, _, _ = summarize_threshold_tuning_strategy(
            y_true=y_true,
            y_prob=y_prob,
            fold_ids=fold_ids,
            target_cols=target_cols,
            selections=selections_by_strategy[strategy],
            baseline_per_label_rows=baseline_per_label_rows,
            baseline_metrics=baseline_metrics,
        )
        report["strategies"][strategy] = strategy_report
        tuned_predictions[strategy] = y_pred

    return report, tuned_predictions


def add_threshold_tuning_columns(
    per_label_df: pd.DataFrame,
    *,
    threshold_tuning_report: Mapping[str, Any],
) -> pd.DataFrame:
    if not threshold_tuning_report.get("enabled", False):
        return per_label_df

    out = per_label_df.copy()
    out["f1_baseline"] = out["f1"]
    for strategy, strategy_report in threshold_tuning_report["strategies"].items():
        by_label = {row["label"]: row for row in strategy_report["per_label_f1"]}
        out[f"f1_tuned_{strategy}"] = [
            by_label[label]["tuned_f1"] for label in out["label"].tolist()
        ]
        out[f"delta_f1_tuned_{strategy}"] = [
            by_label[label]["delta_f1"] for label in out["label"].tolist()
        ]
        out[f"selected_thresholds_{strategy}_by_fold"] = [
            json.dumps(by_label[label]["selected_thresholds_by_fold"], sort_keys=True)
            for label in out["label"].tolist()
        ]
    return out


def threshold_tuning_log_lines(threshold_tuning_report: Mapping[str, Any]) -> List[str]:
    if not threshold_tuning_report.get("enabled", False):
        return ["threshold_tuning_enabled=false"]

    lines = [
        "threshold_tuning_enabled=true",
        f"threshold_tuning_selection_metric={threshold_tuning_report['selection_metric']}",
        f"threshold_tuning_grid={json.dumps(threshold_tuning_report['grid'])}",
        f"threshold_tuning_baseline_threshold={threshold_tuning_report['baseline_threshold']:.6f}",
        f"threshold_tuning_baseline_macro_f1={threshold_tuning_report['baseline_macro_f1']:.6f}",
    ]
    for strategy, strategy_report in threshold_tuning_report["strategies"].items():
        lines.append(
            f"threshold_tuning_{strategy}_macro_f1={strategy_report['tuned_macro_f1']:.6f}"
        )
        lines.append(
            f"threshold_tuning_{strategy}_selected_thresholds="
            f"{json.dumps(strategy_report['selected_thresholds_by_fold'], sort_keys=True)}"
        )
    return lines
