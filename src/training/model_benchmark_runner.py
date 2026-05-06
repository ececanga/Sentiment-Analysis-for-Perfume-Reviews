"""Shared utilities for official classical model benchmark runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

from src.utils.paths import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/codex-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold

from src.training.threshold_tuning import (
    add_threshold_tuning_columns,
    build_inner_validation_manifest,
    build_threshold_tuning_report,
    load_or_write_inner_validation_manifest,
    load_threshold_tuning_settings,
    select_fold_thresholds,
    threshold_tuning_log_lines,
)

CLASSICAL_MODEL_IDS = ("logistic_regression", "xgboost", "random_forest")

ModelBuilder = Callable[[int, np.ndarray], Any]


@dataclass(frozen=True)
class BenchmarkModelSpec:
    model_id: str
    model_family: str
    build_model: ModelBuilder
    model_params: Mapping[str, Any]
    package_names: Sequence[str]


def _metric_or_nan(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid benchmark config at {config_path}")
    return data


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def select_feature_columns(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    cols: List[str] = []
    prefix_tuple = tuple(prefixes)
    for col in df.columns:
        if not col.startswith(prefix_tuple):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def validate_dataset(
    df_raw: pd.DataFrame,
    *,
    unit_id_col: str,
    target_cols: List[str],
    excluded_target_cols: Mapping[str, str],
) -> pd.DataFrame:
    df = df_raw.copy()

    if unit_id_col not in df.columns:
        raise ValueError(f"Missing required unit id column: {unit_id_col}")
    if df[unit_id_col].isna().any():
        raise ValueError(f"Unit id column contains missing values: {unit_id_col}")
    if not df[unit_id_col].astype(str).is_unique:
        raise ValueError(f"Unit id column must be unique: {unit_id_col}")

    required_targets = list(target_cols) + list(excluded_target_cols.keys())
    for col in required_targets:
        if col not in df.columns:
            raise ValueError(f"Missing target column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
        invalid = df[col].isna() | ~df[col].isin([0, 1])
        if invalid.any():
            raise ValueError(f"Target column must be binary 0/1: {col}")

    return df.sort_values(unit_id_col).reset_index(drop=True)


def build_fold_manifest(
    df: pd.DataFrame,
    *,
    unit_id_col: str,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    fold_ids = np.zeros(len(df), dtype=int)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_id, (_, test_idx) in enumerate(splitter.split(df), start=1):
        fold_ids[test_idx] = fold_id

    out = pd.DataFrame(
        {
            unit_id_col: df[unit_id_col].astype(str),
            "fold_id": fold_ids,
        }
    )
    if "Perfume" in df.columns:
        out["Perfume"] = df["Perfume"]
    return out


def load_or_create_fold_manifest(
    *,
    df: pd.DataFrame,
    unit_id_col: str,
    n_splits: int,
    seed: int,
    path: Path,
) -> pd.DataFrame:
    expected_ids = df[unit_id_col].astype(str).tolist()

    if path.exists():
        manifest = pd.read_csv(path)
        if unit_id_col not in manifest.columns or "fold_id" not in manifest.columns:
            raise ValueError(f"Existing fold manifest is missing required columns: {path}")
        if manifest[unit_id_col].astype(str).tolist() != expected_ids:
            raise ValueError(f"Existing fold manifest does not match sorted dataset order: {path}")

        fold_ids = pd.to_numeric(manifest["fold_id"], errors="raise").astype(int)
        expected_fold_ids = set(range(1, n_splits + 1))
        if set(fold_ids.unique().tolist()) != expected_fold_ids:
            raise ValueError(f"Existing fold manifest does not contain folds {sorted(expected_fold_ids)}: {path}")
        manifest["fold_id"] = fold_ids
        return manifest

    manifest = build_fold_manifest(df, unit_id_col=unit_id_col, n_splits=n_splits, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return manifest


def collect_package_versions(extra_packages: Sequence[str] = ()) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in ["numpy", "pandas", "scikit-learn", "matplotlib", "PyYAML", *extra_packages]:
        try:
            versions[name] = version(name)
        except Exception:
            versions[name] = "unknown"
    return versions


def get_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def save_per_label_plot(per_label_df: pd.DataFrame, *, model_family: str, path: Path) -> None:
    labels = per_label_df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, per_label_df["f1"], width=width, label="F1")
    ax.bar(x, per_label_df["roc_auc"], width=width, label="ROC-AUC")
    ax.bar(x + width, per_label_df["pr_auc"], width=width, label="PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_family} Per-Label Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_curve_plot(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str],
    model_family: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, label in enumerate(labels):
        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        precision, recall, _ = precision_recall_curve(yt, yp)
        axes[0].plot(fpr, tpr, linewidth=1.5, label=label)
        axes[1].plot(recall, precision, linewidth=1.5, label=label)

    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    axes[0].set_title(f"{model_family} ROC Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    axes[1].set_title(f"{model_family} Precision-Recall Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_interpretation_note(
    *,
    model_family: str,
    per_label_df: pd.DataFrame,
    global_metrics: Mapping[str, float],
) -> str:
    strongest = per_label_df.sort_values("f1", ascending=False)["label"].head(2).tolist()
    rare = per_label_df.loc[per_label_df["support_pos"] <= 15, "label"].tolist()
    strong_text = ", ".join(strongest) if strongest else "the supported labels"
    rare_text = ", ".join(rare) if rare else "the rare labels"
    return (
        f"{model_family} completed under the frozen 7-label perfume benchmark with macro-F1 "
        f"{global_metrics['macro_f1']:.4f}, macro ROC-AUC {global_metrics['macro_roc_auc']:.4f}, and "
        f"macro PR-AUC {global_metrics['macro_pr_auc']:.4f}. "
        f"It is strongest on {strong_text}, while {rare_text} remain unstable because positive support is very limited. "
        f"The row is benchmark-ready for classical model comparison."
    )


def update_comparison_table(
    *,
    output_path: Path,
    template_path: Path,
    model_id: str,
    model_family: str,
    dataset_relpath: str,
    target_columns: List[str],
    split_design: str,
    seed: int,
    threshold: float,
    metrics: Mapping[str, float],
    note_relpath: str,
) -> None:
    if output_path.exists():
        table_df = pd.read_csv(output_path)
    else:
        table_df = pd.read_csv(template_path)

    row_mask = table_df["model_id"] == model_id
    if not row_mask.any():
        raise ValueError(f"Missing {model_id} row in comparison table template")

    values = {
        "model_family": model_family,
        "dataset": dataset_relpath,
        "granularity": "perfume_level",
        "target_columns": ";".join(target_columns),
        "split_design": split_design,
        "data_seed": seed,
        "model_seed": seed,
        "threshold": threshold,
        "macro_f1": round(metrics["macro_f1"], 6),
        "macro_roc_auc": round(metrics["macro_roc_auc"], 6),
        "macro_pr_auc": round(metrics["macro_pr_auc"], 6),
        "reproducible_run": True,
        "saved_metrics": True,
        "comparable_plots": True,
        "interpretation_note": True,
        "row_complete": True,
        "note_path": note_relpath,
    }
    for col, value in values.items():
        table_df.loc[row_mask, col] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(output_path, index=False)


def write_classical_model_summary(
    *,
    comparison_table_path: Path,
    out_table_path: Path,
    out_figure_path: Path,
) -> None:
    comparison_df = pd.read_csv(comparison_table_path)
    summary_df = comparison_df.loc[
        comparison_df["model_id"].isin(CLASSICAL_MODEL_IDS),
        [
            "model_id",
            "model_family",
            "macro_f1",
            "macro_roc_auc",
            "macro_pr_auc",
            "reproducible_run",
            "saved_metrics",
            "comparable_plots",
            "interpretation_note",
            "row_complete",
            "note_path",
        ],
    ].copy()

    for metric in ["macro_f1", "macro_roc_auc", "macro_pr_auc"]:
        summary_df[metric] = pd.to_numeric(summary_df[metric], errors="coerce")

    out_table_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_table_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = summary_df.loc[
        summary_df[["macro_f1", "macro_roc_auc", "macro_pr_auc"]].notna().any(axis=1)
    ].reset_index(drop=True)

    if plot_df.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No completed classical benchmark rows yet.",
            ha="center",
            va="center",
            fontsize=12,
        )
    else:
        x = np.arange(len(plot_df))
        width = 0.25
        ax.bar(x - width, plot_df["macro_f1"], width=width, label="Macro-F1")
        ax.bar(x, plot_df["macro_roc_auc"], width=width, label="Macro ROC-AUC")
        ax.bar(x + width, plot_df["macro_pr_auc"], width=width, label="Macro PR-AUC")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["model_family"], rotation=15, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("Classical Model Benchmark Summary")
        ax.legend()

    fig.tight_layout()
    out_figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_classical_model_benchmark(
    *,
    input_csv: Path,
    out_root: Path,
    config_path: Path,
    model_spec: BenchmarkModelSpec,
) -> Dict[str, Any]:
    config = load_config(config_path)
    target_cols = list(config["targets"]["official_emotion_columns"])
    excluded_target_cols = dict(config["targets"]["excluded_emotion_columns"])
    feature_prefixes = list(config["data"]["feature_prefixes"])
    unit_id_col = str(config["data"]["unit_id_column"])
    n_splits = int(config["splits"]["outer_cv"]["n_splits"])
    seed = int(config["seeds"]["data_seed"])
    threshold = float(config["evaluation"]["probability_threshold"])
    threshold_tuning_settings = load_threshold_tuning_settings(config, baseline_threshold=threshold)
    inner_cfg = config.get("splits", {}).get("inner_validation", {})
    inner_n_splits = int(inner_cfg.get("n_splits", 10)) if isinstance(inner_cfg, Mapping) else 10

    metrics_dir = out_root / "metrics"
    figures_dir = out_root / "figures"
    predictions_dir = out_root / "predictions"
    logs_dir = out_root / "logs"
    tables_dir = out_root / "tables"
    for path in [metrics_dir, figures_dir, predictions_dir, logs_dir, tables_dir]:
        path.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_csv)
    df = validate_dataset(
        df_raw,
        unit_id_col=unit_id_col,
        target_cols=target_cols,
        excluded_target_cols=excluded_target_cols,
    )
    feature_cols = select_feature_columns(df, feature_prefixes)
    if not feature_cols:
        raise ValueError("No numeric benchmark feature columns found.")

    fold_manifest_path = tables_dir / "fold_assignments.csv"
    fold_manifest = load_or_create_fold_manifest(
        df=df,
        unit_id_col=unit_id_col,
        n_splits=n_splits,
        seed=seed,
        path=fold_manifest_path,
    )

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_cols].to_numpy(dtype=int)
    fold_ids = fold_manifest["fold_id"].to_numpy(dtype=int)
    unit_ids = df[unit_id_col].astype(str).to_numpy()
    inner_manifest_path = tables_dir / f"{model_spec.model_id}_inner_val_folds.csv"
    inner_manifest_df: pd.DataFrame | None = None
    if threshold_tuning_settings.enabled:
        manifest_input = pd.DataFrame(
            {
                unit_id_col: unit_ids,
                "fold_id": fold_ids,
            }
        )
        display_name_col = "Perfume" if "Perfume" in df.columns else None
        if display_name_col is not None:
            manifest_input[display_name_col] = df[display_name_col].astype(str).to_numpy()
        expected_inner_manifest = build_inner_validation_manifest(
            manifest_input,
            unit_id_col=unit_id_col,
            fold_id_col="fold_id",
            display_name_col=display_name_col,
            inner_n_splits=inner_n_splits,
            seed=seed,
        )
        inner_manifest_df = load_or_write_inner_validation_manifest(
            expected_inner_manifest,
            output_path=inner_manifest_path,
        )

    oof_prob = np.zeros((len(df), len(target_cols)), dtype=float)
    oof_pred = np.zeros((len(df), len(target_cols)), dtype=int)
    training_status: Dict[str, List[str]] = {label: [] for label in target_cols}
    fold_threshold_selections: Dict[str, List[Dict[str, Any]]] = {
        strategy: [] for strategy in threshold_tuning_settings.strategies
    }

    for fold_id in sorted(fold_manifest["fold_id"].unique()):
        train_idx = np.where(fold_ids != fold_id)[0]
        test_idx = np.where(fold_ids == fold_id)[0]
        X_train = X[train_idx]
        X_test = X[test_idx]

        if threshold_tuning_settings.enabled:
            if inner_manifest_df is None:
                raise RuntimeError("Threshold tuning is enabled but no inner validation manifest was loaded.")
            inner_fold_df = inner_manifest_df.loc[inner_manifest_df["outer_fold_id"] == fold_id]
            inner_train_ids = set(inner_fold_df.loc[inner_fold_df["inner_role"] == "train", unit_id_col].astype(str))
            val_ids = set(inner_fold_df.loc[inner_fold_df["inner_role"] == "val", unit_id_col].astype(str))
            if not inner_train_ids or not val_ids:
                raise ValueError(f"Inner validation split is empty for outer fold {fold_id}")

            inner_train_idx = np.where(np.isin(unit_ids, list(inner_train_ids)))[0]
            val_idx = np.where(np.isin(unit_ids, list(val_ids)))[0]
            val_prob = np.zeros((len(val_idx), len(target_cols)), dtype=float)

            for label_idx, label in enumerate(target_cols):
                y_inner_train = y[inner_train_idx, label_idx]
                if len(np.unique(y_inner_train)) < 2:
                    const_val = int(y_inner_train[0]) if len(y_inner_train) else 0
                    prob = np.full(len(val_idx), float(const_val))
                else:
                    threshold_model = model_spec.build_model(seed, y_inner_train)
                    threshold_model.fit(X[inner_train_idx], y_inner_train)
                    prob = threshold_model.predict_proba(X[val_idx])[:, 1]
                val_prob[:, label_idx] = prob

            for strategy in threshold_tuning_settings.strategies:
                fold_threshold_selections[strategy].append(
                    select_fold_thresholds(
                        y_val=y[val_idx],
                        prob_val=val_prob,
                        target_cols=target_cols,
                        fold_id=int(fold_id),
                        strategy=strategy,
                        settings=threshold_tuning_settings,
                    )
                )

        for label_idx, label in enumerate(target_cols):
            y_train = y[train_idx, label_idx]
            if len(np.unique(y_train)) < 2:
                const_val = int(y_train[0]) if len(y_train) else 0
                prob = np.full(len(test_idx), float(const_val))
                pred = np.full(len(test_idx), const_val, dtype=int)
                status = f"constant_train_fold_{fold_id}"
            else:
                model = model_spec.build_model(seed, y_train)
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]
                pred = (prob >= threshold).astype(int)
                status = f"trained_fold_{fold_id}"

            oof_prob[test_idx, label_idx] = prob
            oof_pred[test_idx, label_idx] = pred
            training_status[label].append(status)

    per_label_rows: List[Dict[str, Any]] = []
    for label_idx, label in enumerate(target_cols):
        yt = y[:, label_idx]
        yp = oof_pred[:, label_idx]
        yp_prob = oof_prob[:, label_idx]
        per_label_rows.append(
            {
                "label": label,
                "support_pos": int(yt.sum()),
                "support_neg": int(len(yt) - yt.sum()),
                "f1": _metric_or_nan(f1_score, yt, yp, zero_division=0),
                "roc_auc": _metric_or_nan(roc_auc_score, yt, yp_prob),
                "pr_auc": _metric_or_nan(average_precision_score, yt, yp_prob),
                "training_status": ";".join(training_status[label]),
            }
        )

    per_label_df = pd.DataFrame(per_label_rows)
    global_metrics = {
        "macro_f1": float(per_label_df["f1"].mean()),
        "macro_roc_auc": float(per_label_df["roc_auc"].mean()),
        "macro_pr_auc": float(per_label_df["pr_auc"].mean()),
        "micro_f1": _metric_or_nan(f1_score, y, oof_pred, average="micro", zero_division=0),
        "subset_accuracy": float(np.mean(np.all(y == oof_pred, axis=1))),
        "hamming_loss": _metric_or_nan(hamming_loss, y, oof_pred),
        "label_cardinality": float(np.mean(np.sum(y, axis=1))),
    }
    threshold_tuning_report, tuned_predictions_by_strategy = build_threshold_tuning_report(
        settings=threshold_tuning_settings,
        y_true=y,
        y_prob=oof_prob,
        fold_ids=fold_ids,
        target_cols=target_cols,
        selections_by_strategy=fold_threshold_selections,
        baseline_per_label_rows=per_label_rows,
        baseline_metrics=global_metrics,
    )
    per_label_df = add_threshold_tuning_columns(
        per_label_df,
        threshold_tuning_report=threshold_tuning_report,
    )

    oof_df = fold_manifest.copy()
    if "Perfume" in df.columns and "Perfume" not in oof_df.columns:
        oof_df["Perfume"] = df["Perfume"]
    for label_idx, label in enumerate(target_cols):
        oof_df[f"true_{label}"] = y[:, label_idx]
        oof_df[f"prob_{label}"] = oof_prob[:, label_idx]
        oof_df[f"pred_{label}"] = oof_pred[:, label_idx]
        for strategy, tuned_pred in tuned_predictions_by_strategy.items():
            oof_df[f"pred_tuned_{strategy}_{label}"] = tuned_pred[:, label_idx]

    predictions_path = predictions_dir / f"{model_spec.model_id}_oof.csv"
    per_label_path = metrics_dir / f"{model_spec.model_id}_per_label.csv"
    metrics_path = metrics_dir / f"{model_spec.model_id}.json"
    per_label_plot_path = figures_dir / f"{model_spec.model_id}_per_label_metrics.png"
    curve_plot_path = figures_dir / f"{model_spec.model_id}_roc_pr_curves.png"
    log_path = logs_dir / f"{model_spec.model_id}.log"
    run_manifest_path = logs_dir / f"{model_spec.model_id}_run.json"
    note_path = tables_dir / f"{model_spec.model_id}_note.md"
    comparison_table_path = tables_dir / "comparison_table.csv"
    comparison_template_path = PROJECT_ROOT / "docs" / "comparison_table_template.csv"
    classical_summary_table_path = tables_dir / "classical_model_comparison.csv"
    classical_summary_figure_path = figures_dir / "classical_model_comparison.png"

    oof_df.to_csv(predictions_path, index=False)
    per_label_df.to_csv(per_label_path, index=False)
    save_per_label_plot(per_label_df, model_family=model_spec.model_family, path=per_label_plot_path)
    save_curve_plot(
        y_true=y,
        y_prob=oof_prob,
        labels=target_cols,
        model_family=model_spec.model_family,
        path=curve_plot_path,
    )

    note_text = build_interpretation_note(
        model_family=model_spec.model_family,
        per_label_df=per_label_df,
        global_metrics=global_metrics,
    )
    note_path.write_text(note_text + "\n", encoding="utf-8")

    timestamp_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    dataset_relpath = stringify_path(input_csv)
    fold_manifest_relpath = stringify_path(fold_manifest_path)
    inner_manifest_relpath = stringify_path(inner_manifest_path) if threshold_tuning_settings.enabled else None
    artifact_paths = {
        "metrics_json": stringify_path(metrics_path),
        "per_label_csv": stringify_path(per_label_path),
        "predictions_csv": stringify_path(predictions_path),
        "per_label_plot": stringify_path(per_label_plot_path),
        "roc_pr_plot": stringify_path(curve_plot_path),
        "log": stringify_path(log_path),
        "run_manifest": stringify_path(run_manifest_path),
        "note": stringify_path(note_path),
        "comparison_table": stringify_path(comparison_table_path),
        "fold_manifest": fold_manifest_relpath,
        "classical_summary_table": stringify_path(classical_summary_table_path),
        "classical_summary_figure": stringify_path(classical_summary_figure_path),
    }
    if inner_manifest_relpath is not None:
        artifact_paths["inner_validation_manifest"] = inner_manifest_relpath
    run_id = f"{model_spec.model_id}_seed{seed}_kfold{n_splits}"

    metrics_payload: Dict[str, Any] = {
        "model_id": model_spec.model_id,
        "model_family": model_spec.model_family,
        "run_id": run_id,
        "dataset_path": dataset_relpath,
        "n_rows": int(len(df)),
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "excluded_target_columns": excluded_target_cols,
        "split_manifest_path": fold_manifest_relpath,
        "inner_validation_manifest_path": inner_manifest_relpath,
        "data_seed": seed,
        "model_seed": seed,
        "threshold": threshold,
        "baseline_threshold": threshold,
        "model_parameters": dict(model_spec.model_params),
        "macro_f1": global_metrics["macro_f1"],
        "macro_roc_auc": global_metrics["macro_roc_auc"],
        "macro_pr_auc": global_metrics["macro_pr_auc"],
        "secondary_metrics": {
            "micro_f1": global_metrics["micro_f1"],
            "subset_accuracy": global_metrics["subset_accuracy"],
            "hamming_loss": global_metrics["hamming_loss"],
            "label_cardinality": global_metrics["label_cardinality"],
        },
        "per_label": per_label_rows,
        "threshold_tuning": threshold_tuning_report,
        "artifact_paths": artifact_paths,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    run_manifest = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "package_versions": collect_package_versions(model_spec.package_names),
        "dataset_path": dataset_relpath,
        "split_manifest_path": fold_manifest_relpath,
        "inner_validation_manifest_path": inner_manifest_relpath,
        "data_seed": seed,
        "model_seed": seed,
        "threshold": threshold,
        "baseline_threshold": threshold,
        "command": " ".join(sys.argv),
        "device": "cpu",
        "model_id": model_spec.model_id,
        "model_family": model_spec.model_family,
        "model_parameters": dict(model_spec.model_params),
        "threshold_tuning": threshold_tuning_report,
        "artifact_paths": artifact_paths,
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    log_lines = [
        f"timestamp_utc={timestamp_utc}",
        f"model_id={model_spec.model_id}",
        f"model_family={model_spec.model_family}",
        f"dataset={dataset_relpath}",
        f"n_rows={len(df)}",
        f"n_features={len(feature_cols)}",
        f"fold_manifest={fold_manifest_relpath}",
        f"inner_validation_manifest={inner_manifest_relpath}",
        f"macro_f1={global_metrics['macro_f1']:.6f}",
        f"macro_roc_auc={global_metrics['macro_roc_auc']:.6f}",
        f"macro_pr_auc={global_metrics['macro_pr_auc']:.6f}",
        f"micro_f1={global_metrics['micro_f1']:.6f}",
        f"subset_accuracy={global_metrics['subset_accuracy']:.6f}",
        f"hamming_loss={global_metrics['hamming_loss']:.6f}",
        f"model_parameters={json.dumps(model_spec.model_params, sort_keys=True)}",
    ]
    log_lines.extend(threshold_tuning_log_lines(threshold_tuning_report))
    for row in per_label_rows:
        log_lines.append(
            "label="
            + row["label"]
            + f" support_pos={row['support_pos']} f1={row['f1']:.6f} roc_auc={row['roc_auc']:.6f} pr_auc={row['pr_auc']:.6f}"
        )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    update_comparison_table(
        output_path=comparison_table_path,
        template_path=comparison_template_path,
        model_id=model_spec.model_id,
        model_family=model_spec.model_family,
        dataset_relpath=dataset_relpath,
        target_columns=target_cols,
        split_design=f"{n_splits}-fold KFold",
        seed=seed,
        threshold=threshold,
        metrics=global_metrics,
        note_relpath=stringify_path(note_path),
    )
    write_classical_model_summary(
        comparison_table_path=comparison_table_path,
        out_table_path=classical_summary_table_path,
        out_figure_path=classical_summary_figure_path,
    )

    return {
        "metrics_path": metrics_path,
        "per_label_path": per_label_path,
        "predictions_path": predictions_path,
        "fold_manifest_path": fold_manifest_path,
        "inner_validation_manifest_path": inner_manifest_path if threshold_tuning_settings.enabled else None,
        "comparison_table_path": comparison_table_path,
        "note_path": note_path,
        "per_label_plot_path": per_label_plot_path,
        "curve_plot_path": curve_plot_path,
        "run_manifest_path": run_manifest_path,
        "log_path": log_path,
        "classical_summary_table_path": classical_summary_table_path,
        "classical_summary_figure_path": classical_summary_figure_path,
        "macro_metrics": global_metrics,
        "threshold_tuning": threshold_tuning_report,
    }
