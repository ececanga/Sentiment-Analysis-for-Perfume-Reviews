"""train_eval_gold43.py

Train/evaluate baseline models on the 43 labeled perfumes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.paths import data_path, outputs_path

EMOTIONS = ["joy", "sadness", "anger", "fear", "trust", "surprise", "anticipation", "disgust"]
VAD = ["valence", "arousal", "dominance"]
SEEDS = [0, 1, 2, 3, 4]


def normalize_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def find_perfume_column(columns: Iterable[str]) -> str:
    matches = [c for c in columns if c.strip().lower() == "perfume"]
    if not matches:
        raise ValueError("No perfume column found. Expected 'Perfume' (case-insensitive).")
    return matches[0]


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    x_rank = pd.Series(x).rank().to_numpy()
    y_rank = pd.Series(y).rank().to_numpy()
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def evaluate_emotions_fold(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    per_label_f1: Dict[str, float | None] = {}
    keep_idx: List[int] = []
    for i, emo in enumerate(EMOTIONS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        is_const_true = np.all(yt == 0) or np.all(yt == 1)
        is_const_pred = np.all(yp == 0) or np.all(yp == 1)
        if is_const_true and is_const_pred:
            per_label_f1[emo] = None
            continue
        per_label_f1[emo] = float(f1_score(yt, yp, average="binary", zero_division=0))
        keep_idx.append(i)

    if keep_idx:
        macro_f1 = float(np.mean([per_label_f1[EMOTIONS[i]] for i in keep_idx]))
        micro_f1 = float(f1_score(y_true[:, keep_idx], y_pred[:, keep_idx], average="micro", zero_division=0))
    else:
        macro_f1 = float("nan")
        micro_f1 = float("nan")

    return {
        "per_label_f1": per_label_f1,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "label_cardinality": float(y_true.sum(axis=1).mean()),
    }


def evaluate_vad(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for i, dim in enumerate(VAD):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = math.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        rho = spearman_corr(y_true[:, i], y_pred[:, i])
        metrics[dim] = {"mae": float(mae), "rmse": float(rmse), "spearman": float(rho)}
    return metrics


def build_emotion_model() -> Pipeline:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", OneVsRestClassifier(clf)),
    ])


def build_vad_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge()),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labels_csv",
        type=Path,
        default=data_path("annotation_packets", "Labeled_Annotation_Packet_strict_v1.csv"),
        help="Labeled packet CSV path",
    )
    ap.add_argument(
        "--features_csv",
        type=Path,
        default=data_path("perfume_level", "Unified_Emotion_Profile_Perfume.csv"),
        help="Unified perfume-level feature CSV",
    )
    ap.add_argument(
        "--out_json",
        type=Path,
        default=outputs_path("training", "gold43_baseline_metrics.json"),
        help="Output JSON summary path",
    )
    ap.add_argument(
        "--out_oof",
        type=Path,
        default=outputs_path("training", "gold43_oof_predictions.csv"),
        help="Output OOF predictions CSV",
    )
    args = ap.parse_args()

    labels_csv = args.labels_csv
    features_csv = args.features_csv

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    labels_df = pd.read_csv(labels_csv)
    features_df = pd.read_csv(features_csv)

    label_perfume_col = find_perfume_column(labels_df.columns)
    feat_perfume_col = find_perfume_column(features_df.columns)

    labels_df["_perfume_norm"] = labels_df[label_perfume_col].apply(normalize_name)
    features_df["_perfume_norm"] = features_df[feat_perfume_col].apply(normalize_name)

    labels_norm = set(labels_df["_perfume_norm"].dropna().unique())
    features_norm = set(features_df["_perfume_norm"].dropna().unique())

    missing_in_features = sorted(labels_norm.difference(features_norm))
    extra_in_features = sorted(features_norm.difference(labels_norm))

    if missing_in_features:
        print(f"Missing features for {len(missing_in_features)} perfumes (first 10): {missing_in_features[:10]}")
    if extra_in_features:
        print(f"Extra features (no labels) for {len(extra_in_features)} perfumes (first 10): {extra_in_features[:10]}")

    merged = pd.merge(labels_df, features_df, on="_perfume_norm", how="inner", suffixes=("", "_feat"))

    for col in EMOTIONS + VAD:
        if col not in merged.columns:
            raise ValueError(f"Missing required label column: {col}")

    for emo in EMOTIONS:
        parsed = pd.to_numeric(merged[emo], errors="coerce")
        invalid = parsed.isna() | ~parsed.isin([0, 1])
        if invalid.any():
            names = merged.loc[invalid, label_perfume_col].astype(str).unique().tolist()
            raise ValueError(f"Invalid values for emotion '{emo}' in perfumes: {names}")
        merged[emo] = parsed.astype(int)

    for dim in VAD:
        merged[dim] = pd.to_numeric(merged[dim], errors="coerce")

    vad_missing_mask = merged[VAD].isna().any(axis=1)
    vad_missing_count = int(vad_missing_mask.sum())
    if vad_missing_count:
        missing_rows = merged.loc[vad_missing_mask, [label_perfume_col] + VAD]
        lines = []
        for _, row in missing_rows.iterrows():
            missing_dims = [d for d in VAD if pd.isna(row[d])]
            lines.append(f\"{row[label_perfume_col]}: {missing_dims}\")
        print(\"VAD missing values detected:\")
        for line in lines:
            print(f\"  - {line}\")

    df_vad = merged.dropna(subset=VAD)

    feature_cols = [c for c in features_df.columns if c != feat_perfume_col and c != "_perfume_norm"]
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(features_df[c])]
    if not numeric_features:
        raise ValueError("No numeric feature columns found in unified profile CSV.")

    X = merged[numeric_features].to_numpy(dtype=float)
    y_emotions = merged[EMOTIONS].to_numpy(dtype=int)
    y_vad = df_vad[VAD].to_numpy(dtype=float)
    X_vad = df_vad[numeric_features].to_numpy(dtype=float)
    vad_indices = df_vad.index.to_numpy()

    n_samples = X.shape[0]
    n_samples_vad = y_vad.shape[0]
    oof_prob_sum = np.zeros((n_samples, len(EMOTIONS)), dtype=float)
    oof_vad_sum = np.zeros((n_samples, len(VAD)), dtype=float)
    oof_vad_counts = np.zeros(n_samples, dtype=float)

    emotion_metrics_by_seed: List[Dict[str, object]] = []
    vad_metrics_by_seed: List[Dict[str, Dict[str, float]]] = []
    degenerate_labels_by_fold: List[Dict[str, object]] = []

    for seed in SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_probs = np.zeros((n_samples, len(EMOTIONS)), dtype=float)
        oof_vad = np.zeros((n_samples, len(VAD)), dtype=float)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train_em = y_emotions[train_idx]

            degenerate = []
            for i, emo in enumerate(EMOTIONS):
                col = y_train_em[:, i]
                if np.all(col == 0) or np.all(col == 1):
                    degenerate.append(emo)
            degenerate_labels_by_fold.append(
                {"seed": seed, "fold": fold_idx, "degenerate_labels": degenerate}
            )

            em_model = build_emotion_model()
            em_model.fit(X_train, y_train_em)
            probs = em_model.predict_proba(X_test)
            oof_probs[test_idx] = probs

            y_test_em = y_emotions[test_idx]
            y_pred_em = (probs >= 0.5).astype(int)
            emotion_metrics_by_seed.append(evaluate_emotions_fold(y_test_em, y_pred_em))

        if n_samples_vad:
            kf_vad = KFold(n_splits=5, shuffle=True, random_state=seed)
            oof_vad_seed = np.zeros((n_samples_vad, len(VAD)), dtype=float)
            for train_idx, test_idx in kf_vad.split(X_vad):
                X_train, X_test = X_vad[train_idx], X_vad[test_idx]
                y_train_vad = y_vad[train_idx]

                vad_model = build_vad_model()
                vad_model.fit(X_train, y_train_vad)
                vad_pred = vad_model.predict(X_test)
                oof_vad_seed[test_idx] = vad_pred

            for i, row_idx in enumerate(vad_indices):
                oof_vad[row_idx] = oof_vad_seed[i]
                oof_vad_counts[row_idx] += 1.0
            vad_metrics_by_seed.append(evaluate_vad(y_vad, oof_vad_seed))

        oof_prob_sum += oof_probs
        oof_vad_sum += oof_vad

    avg_probs = oof_prob_sum / len(SEEDS)
    avg_vad = np.full((n_samples, len(VAD)), np.nan, dtype=float)
    for i in range(n_samples):
        if oof_vad_counts[i] > 0:
            avg_vad[i] = oof_vad_sum[i] / oof_vad_counts[i]
    avg_pred_em = (avg_probs >= 0.5).astype(int)

    per_label_f1_sum = {emo: 0.0 for emo in EMOTIONS}
    per_label_f1_count = {emo: 0 for emo in EMOTIONS}
    for metrics in emotion_metrics_by_seed:
        for emo, score in metrics["per_label_f1"].items():
            if score is None:
                continue
            per_label_f1_sum[emo] += float(score)
            per_label_f1_count[emo] += 1
    per_label_f1 = {
        emo: (per_label_f1_sum[emo] / per_label_f1_count[emo]) if per_label_f1_count[emo] else float("nan")
        for emo in EMOTIONS
    }

    macro_f1 = float(np.nanmean([m["macro_f1"] for m in emotion_metrics_by_seed]))
    micro_f1 = float(np.nanmean([m["micro_f1"] for m in emotion_metrics_by_seed]))
    subset_acc = float(np.nanmean([m["subset_accuracy"] for m in emotion_metrics_by_seed]))
    label_card = float(np.nanmean([m["label_cardinality"] for m in emotion_metrics_by_seed]))

    vad_avg_metrics: Dict[str, Dict[str, float]] = {dim: {"mae": 0.0, "rmse": 0.0, "spearman": 0.0} for dim in VAD}
    for metrics in vad_metrics_by_seed:
        for dim in VAD:
            for k in vad_avg_metrics[dim]:
                vad_avg_metrics[dim][k] += metrics[dim][k]
    for dim in VAD:
        for k in vad_avg_metrics[dim]:
            vad_avg_metrics[dim][k] /= max(1, len(vad_metrics_by_seed))

    summary = {
        "n_samples": int(n_samples),
        "n_rows_emotions": int(n_samples),
        "n_rows_vad": int(n_samples_vad),
        "n_features": int(X.shape[1]),
        "vad_missing_count": vad_missing_count,
        "emotion_metrics": {
            "per_label_f1": per_label_f1,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "subset_accuracy": subset_acc,
            "label_cardinality": label_card,
        },
        "vad_metrics": vad_avg_metrics,
        "fold_diagnostics": {
            "degenerate_labels_by_fold": degenerate_labels_by_fold,
        },
        "seeds": SEEDS,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_oof.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    oof_df = pd.DataFrame({
        "Perfume": merged[label_perfume_col],
    })
    for i, emo in enumerate(EMOTIONS):
        oof_df[f"true_{emo}"] = y_emotions[:, i]
        oof_df[f"prob_{emo}"] = avg_probs[:, i]
        oof_df[f"pred_{emo}"] = avg_pred_em[:, i]

    for i, dim in enumerate(VAD):
        oof_df[f"true_{dim}"] = merged[dim].to_numpy(dtype=float)
        oof_df[f"pred_{dim}"] = avg_vad[:, i]

    oof_df.to_csv(args.out_oof, index=False)

    print("Gold43 baseline (5x5-fold CV) summary")
    print(f"- Samples: {n_samples}")
    print(f"- Features: {X.shape[1]}")
    print(f"- Emotions macro-F1: {macro_f1:.3f} | micro-F1: {micro_f1:.3f} | subset acc: {subset_acc:.3f}")
    print("- VAD MAE/RMSE/Spearman:")
    for dim in VAD:
        m = vad_avg_metrics[dim]
        print(f"  {dim}: MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} rho={m['spearman']:.3f}")


if __name__ == "__main__":
    main()
