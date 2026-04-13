"""Stage 7 supervised training on merged 784-perfume dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.paths import outputs_path

EMOTION_COLS = ["joy", "sadness", "anger", "fear", "trust", "surprise", "anticipation", "disgust"]
VAD_COLS = ["valence", "arousal", "dominance"]
FEATURE_PREFIXES = ("NRC_", "VAD_", "Hartmann_", "Savani_", "GoEmotions_", "Cardiff_")


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    out = pearsonr(x, y)
    return float(out.statistic if hasattr(out, "statistic") else out[0])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    out = spearmanr(x, y)
    return float(out.statistic if hasattr(out, "statistic") else out[0])


def build_emotion_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def build_vad_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", Ridge()),
        ]
    )


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if not c.startswith(FEATURE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def validate_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in EMOTION_COLS + VAD_COLS:
        if col not in out.columns:
            raise ValueError(f"Missing target column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    bad_emo = {}
    for col in EMOTION_COLS:
        s = out[col]
        invalid = s.isna() | ~s.isin([0, 1])
        if invalid.any():
            bad_emo[col] = int(invalid.sum())
    if bad_emo:
        raise ValueError(f"Invalid binary values in emotion targets: {bad_emo}")
    if out[VAD_COLS].isna().any().any():
        missing = out[VAD_COLS].isna().sum().to_dict()
        raise ValueError(f"Missing VAD target values: {missing}")
    return out


def _metric_or_nan(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def run_training(
    *,
    input_csv: Path,
    out_dir: Path,
    seed: int = 42,
    emotion_threshold: float = 0.5,
    max_folds: int = 5,
    save_models: bool = False,
) -> Dict[str, Any]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_csv)
    df = validate_targets(df_raw)
    feature_cols = select_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found with expected prefixes.")

    X = df[feature_cols].to_numpy(dtype=float)
    y_em = df[EMOTION_COLS].to_numpy(dtype=int)
    y_vad = df[VAD_COLS].to_numpy(dtype=float)
    n = len(df)

    oof_prob_em = np.zeros((n, len(EMOTION_COLS)), dtype=float)
    oof_pred_em = np.zeros((n, len(EMOTION_COLS)), dtype=int)
    per_label_rows: List[Dict[str, Any]] = []
    nondegenerate_idx: List[int] = []

    emotion_status: Dict[str, str] = {}

    for i, emo in enumerate(EMOTION_COLS):
        y = y_em[:, i]
        n_pos = int(y.sum())
        n_neg = int(n - n_pos)
        prevalence = float(n_pos / n) if n else float("nan")
        is_constant = n_pos == 0 or n_neg == 0
        n_splits = 0 if is_constant else int(min(max_folds, n_pos, n_neg))

        if is_constant:
            const_val = int(y[0]) if n else 0
            prob = np.full(n, float(const_val))
            pred = np.full(n, const_val, dtype=int)
            status = "degenerate_constant"
        elif n_splits < 2:
            const_val = int(1 if n_pos >= n_neg else 0)
            prob = np.full(n, float(const_val))
            pred = np.full(n, const_val, dtype=int)
            status = "insufficient_minority"
        else:
            status = "trained"
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            prob = np.zeros(n, dtype=float)
            for tr_idx, te_idx in skf.split(X, y):
                model = build_emotion_model()
                model.fit(X[tr_idx], y[tr_idx])
                prob[te_idx] = model.predict_proba(X[te_idx])[:, 1]
            pred = (prob >= emotion_threshold).astype(int)

        oof_prob_em[:, i] = prob
        oof_pred_em[:, i] = pred
        emotion_status[emo] = status

        if n_pos > 0 and n_neg > 0:
            nondegenerate_idx.append(i)

        row = {
            "emotion": emo,
            "n": int(n),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "prevalence": prevalence,
            "is_constant": bool(is_constant),
            "n_splits": int(n_splits),
            "status": status,
            "precision": _metric_or_nan(precision_score, y, pred, zero_division=0),
            "recall": _metric_or_nan(recall_score, y, pred, zero_division=0),
            "f1": _metric_or_nan(f1_score, y, pred, zero_division=0),
            "balanced_accuracy": float("nan"),
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
        }
        if n_pos > 0 and n_neg > 0:
            row["balanced_accuracy"] = _metric_or_nan(balanced_accuracy_score, y, pred)
            row["roc_auc"] = _metric_or_nan(roc_auc_score, y, prob)
            row["average_precision"] = _metric_or_nan(average_precision_score, y, prob)
        per_label_rows.append(row)

    if nondegenerate_idx:
        macro_f1_nondegenerate = float(
            np.nanmean([per_label_rows[i]["f1"] for i in nondegenerate_idx])
        )
        micro_f1_nondegenerate = _metric_or_nan(
            f1_score,
            y_em[:, nondegenerate_idx],
            oof_pred_em[:, nondegenerate_idx],
            average="micro",
            zero_division=0,
        )
    else:
        macro_f1_nondegenerate = float("nan")
        micro_f1_nondegenerate = float("nan")

    global_emotion_metrics = {
        "macro_f1_nondegenerate": macro_f1_nondegenerate,
        "micro_f1_nondegenerate": micro_f1_nondegenerate,
        "subset_accuracy": _metric_or_nan((lambda a, b: np.mean(np.all(a == b, axis=1))), y_em, oof_pred_em),
        "hamming_loss": _metric_or_nan(hamming_loss, y_em, oof_pred_em),
        "label_cardinality": float(np.mean(np.sum(y_em, axis=1))),
    }

    # VAD OOF
    oof_vad = np.full((n, len(VAD_COLS)), np.nan, dtype=float)
    n_splits_vad = int(min(max_folds, n))
    if n_splits_vad >= 2:
        kf = KFold(n_splits=n_splits_vad, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(X):
            vad_model = build_vad_model()
            vad_model.fit(X[tr_idx], y_vad[tr_idx])
            oof_vad[te_idx] = vad_model.predict(X[te_idx])
        vad_status = "trained"
    else:
        mean_vec = np.mean(y_vad, axis=0)
        oof_vad[:, :] = mean_vec
        vad_status = "insufficient_rows"

    vad_rows: List[Dict[str, Any]] = []
    for i, dim in enumerate(VAD_COLS):
        y_true = y_vad[:, i]
        y_pred = oof_vad[:, i]
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        n_dim = int(mask.sum())
        if n_dim == 0:
            row = {
                "dimension": dim,
                "n": 0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "pearson_r": float("nan"),
                "spearman_r": float("nan"),
            }
        else:
            yt = y_true[mask]
            yp = y_pred[mask]
            row = {
                "dimension": dim,
                "n": n_dim,
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(math.sqrt(mean_squared_error(yt, yp))),
                "r2": _metric_or_nan(r2_score, yt, yp) if n_dim >= 2 else float("nan"),
                "pearson_r": safe_pearson(yt, yp),
                "spearman_r": safe_spearman(yt, yp),
            }
        vad_rows.append(row)

    vad_df = pd.DataFrame(vad_rows)
    avg_row = {
        "dimension": "average",
        "n": int(np.nanmean(vad_df["n"])) if not vad_df.empty else 0,
        "mae": float(np.nanmean(vad_df["mae"])) if not vad_df.empty else float("nan"),
        "rmse": float(np.nanmean(vad_df["rmse"])) if not vad_df.empty else float("nan"),
        "r2": float(np.nanmean(vad_df["r2"])) if not vad_df.empty else float("nan"),
        "pearson_r": float(np.nanmean(vad_df["pearson_r"])) if not vad_df.empty else float("nan"),
        "spearman_r": float(np.nanmean(vad_df["spearman_r"])) if not vad_df.empty else float("nan"),
    }
    vad_df = pd.concat([vad_df, pd.DataFrame([avg_row])], ignore_index=True)

    # Outputs
    metrics_path = out_dir / "stage7_gemini784_metrics.json"
    oof_path = out_dir / "stage7_gemini784_oof_predictions.csv"
    label_diag_path = out_dir / "stage7_gemini784_label_diagnostics.csv"
    vad_metrics_path = out_dir / "stage7_gemini784_vad_metrics.csv"
    model_path = out_dir / "stage7_gemini784_models.joblib"

    label_diag_df = pd.DataFrame(per_label_rows)
    label_diag_df.to_csv(label_diag_path, index=False)
    vad_df.to_csv(vad_metrics_path, index=False)

    id_cols = [c for c in ["perfume_id", "Perfume", "perfume_name_unified", "perfume_name", "perfume_name_gemini"] if c in df.columns]
    oof_df = df[id_cols].copy()
    for i, emo in enumerate(EMOTION_COLS):
        oof_df[f"true_{emo}"] = y_em[:, i]
        oof_df[f"prob_{emo}"] = oof_prob_em[:, i]
        oof_df[f"pred_{emo}"] = oof_pred_em[:, i]
    for i, dim in enumerate(VAD_COLS):
        oof_df[f"true_{dim}"] = y_vad[:, i]
        oof_df[f"pred_{dim}"] = oof_vad[:, i]
    oof_df.to_csv(oof_path, index=False)

    metrics: Dict[str, Any] = {
        "input_csv": str(input_csv),
        "n_rows": int(n),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "config": {
            "seed": seed,
            "emotion_threshold": float(emotion_threshold),
            "max_folds": int(max_folds),
            "save_models": bool(save_models),
        },
        "emotion_metrics": {
            "per_label": per_label_rows,
            "global": global_emotion_metrics,
        },
        "vad_metrics": vad_df.to_dict(orient="records"),
        "diagnostics": {
            "emotion_status": emotion_status,
            "vad_status": vad_status,
            "nondegenerate_labels": [EMOTION_COLS[i] for i in nondegenerate_idx],
        },
        "artifacts": {
            "metrics_json": str(metrics_path),
            "oof_csv": str(oof_path),
            "label_diagnostics_csv": str(label_diag_path),
            "vad_metrics_csv": str(vad_metrics_path),
            "models_joblib": str(model_path) if save_models else None,
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if save_models:
        emotion_models: Dict[str, Any] = {}
        for i, emo in enumerate(EMOTION_COLS):
            y = y_em[:, i]
            status = emotion_status[emo]
            if status == "trained":
                model = build_emotion_model()
                model.fit(X, y)
                emotion_models[emo] = {"type": "model", "model": model}
            else:
                const_val = int(round(np.mean(y)))
                emotion_models[emo] = {"type": "constant", "value": const_val}

        vad_model_full = build_vad_model()
        vad_model_full.fit(X, y_vad)
        model_bundle = {
            "feature_columns": feature_cols,
            "emotion_columns": EMOTION_COLS,
            "vad_columns": VAD_COLS,
            "emotion_models": emotion_models,
            "vad_model": vad_model_full,
            "config": metrics["config"],
        }
        joblib.dump(model_bundle, model_path)

    print("Stage 7 training complete")
    print(f"- Rows: {n}")
    print(f"- Features: {len(feature_cols)}")
    print(f"- OOF: {oof_path}")
    print(f"- Metrics: {metrics_path}")
    print(f"- Label diagnostics: {label_diag_path}")
    print(f"- VAD metrics: {vad_metrics_path}")
    if save_models:
        print(f"- Models: {model_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-csv",
        type=Path,
        default=outputs_path("analysis_out", "gemini_unified_merged.csv"),
        help="Merged Stage 6 CSV path",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=outputs_path("training"),
        help="Output directory for Stage 7 artifacts",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emotion-threshold", type=float, default=0.5)
    ap.add_argument("--max-folds", type=int, default=5)
    ap.add_argument(
        "--save-models",
        action="store_true",
        default=False,
        help="If set, saves optional fitted model bundle (secondary artifact).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        seed=args.seed,
        emotion_threshold=args.emotion_threshold,
        max_folds=args.max_folds,
        save_models=args.save_models,
    )


if __name__ == "__main__":
    main()

