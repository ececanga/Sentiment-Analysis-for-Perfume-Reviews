# Date Created: 2026-04-13 23:02:44+03:00
# Date Last Modified: 2026-05-15

"""Stage 8A: validation-safe permutation importance for Gemini-784 models.

This analysis is table-first and computes:
1) Feature-level permutation importance on held-out folds.
2) Family-level grouped permutation importance on held-out folds.

It reuses Stage 7 artifact metadata for canonical feature selection and
target diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.paths import outputs_path

EMOTION_COLS = ["joy", "sadness", "anger", "fear", "trust", "surprise", "anticipation", "disgust"]
VAD_COLS = ["valence", "arousal", "dominance"]
REQUIRED_FAMILIES = ["NRC", "VAD", "Hartmann", "Savani", "GoEmotions", "Cardiff"]
FAMILY_PREFIXES: List[Tuple[str, str]] = [
    ("NRC_", "NRC"),
    ("VAD_", "VAD"),
    ("Hartmann_", "Hartmann"),
    ("Savani_", "Savani"),
    ("GoEmotions_", "GoEmotions"),
    ("Cardiff_", "Cardiff"),
]


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


def _safe_float(x: Any) -> float:
    try:
        out = float(x)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) < 2 or np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _delta_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "importance_mean": float("nan"),
            "importance_std": float("nan"),
            "importance_median": float("nan"),
        }
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "importance_mean": float("nan"),
            "importance_std": float("nan"),
            "importance_median": float("nan"),
        }
    return {
        "importance_mean": float(np.mean(arr)),
        "importance_std": float(np.std(arr, ddof=0)),
        "importance_median": float(np.median(arr)),
    }


def _metric_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def feature_family(feature_name: str) -> str:
    for prefix, family in FAMILY_PREFIXES:
        if feature_name.startswith(prefix):
            return family
    return "Other"


def build_family_index(feature_cols: List[str]) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[int]]]:
    feature_to_family: Dict[str, str] = {}
    family_to_features: Dict[str, List[str]] = {family: [] for family in REQUIRED_FAMILIES}

    for feat in feature_cols:
        fam = feature_family(feat)
        feature_to_family[feat] = fam
        if fam in family_to_features:
            family_to_features[fam].append(feat)

    feature_idx = {c: i for i, c in enumerate(feature_cols)}
    family_to_indices = {
        fam: [feature_idx[f] for f in feats if f in feature_idx] for fam, feats in family_to_features.items()
    }
    family_to_features = {k: v for k, v in family_to_features.items() if len(v) > 0}
    family_to_indices = {k: v for k, v in family_to_indices.items() if len(v) > 0}
    return feature_to_family, family_to_features, family_to_indices


def _raise_or_warn(strict: bool, message: str) -> None:
    if strict:
        raise ValueError(message)
    print(f"[WARN] {message}")


def validate_and_align_inputs(
    *,
    merged_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    strict: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if "perfume_id" not in merged_df.columns:
        raise ValueError("Merged CSV is missing required column 'perfume_id'.")
    if "perfume_id" not in oof_df.columns:
        raise ValueError("Stage 7 OOF CSV is missing required column 'perfume_id'.")

    report: Dict[str, Any] = {
        "merged_rows": int(len(merged_df)),
        "oof_rows": int(len(oof_df)),
        "strict_mode": bool(strict),
    }

    merged_dups = sorted(
        merged_df.loc[merged_df["perfume_id"].duplicated(keep=False), "perfume_id"].astype(str).unique().tolist()
    )
    oof_dups = sorted(oof_df.loc[oof_df["perfume_id"].duplicated(keep=False), "perfume_id"].astype(str).unique().tolist())
    report["merged_duplicate_perfume_ids"] = merged_dups
    report["oof_duplicate_perfume_ids"] = oof_dups

    if merged_dups:
        _raise_or_warn(strict, f"Merged CSV has duplicate perfume_id values (count={len(merged_dups)}).")
        merged_df = merged_df.drop_duplicates(subset=["perfume_id"], keep="first").copy()
    if oof_dups:
        _raise_or_warn(strict, f"OOF CSV has duplicate perfume_id values (count={len(oof_dups)}).")
        oof_df = oof_df.drop_duplicates(subset=["perfume_id"], keep="first").copy()

    merged_ids = set(merged_df["perfume_id"].astype(str))
    oof_ids = set(oof_df["perfume_id"].astype(str))
    merged_only = sorted(merged_ids - oof_ids)
    oof_only = sorted(oof_ids - merged_ids)
    report["merged_unique_ids"] = int(len(merged_ids))
    report["oof_unique_ids"] = int(len(oof_ids))
    report["merged_only_count"] = int(len(merged_only))
    report["oof_only_count"] = int(len(oof_only))
    report["merged_only_sample"] = merged_only[:20]
    report["oof_only_sample"] = oof_only[:20]

    if merged_only or oof_only:
        _raise_or_warn(
            strict,
            "Mismatch between merged and OOF perfume_id sets "
            f"(merged_only={len(merged_only)}, oof_only={len(oof_only)}).",
        )

    oof_indexed = oof_df.set_index("perfume_id", drop=False)
    aligned_ids = [pid for pid in merged_df["perfume_id"].astype(str).tolist() if pid in oof_indexed.index]
    merged_aligned = merged_df[merged_df["perfume_id"].astype(str).isin(aligned_ids)].copy()
    oof_aligned = oof_indexed.loc[aligned_ids].reset_index(drop=True).copy()

    report["matched_count"] = int(len(aligned_ids))
    return merged_aligned.reset_index(drop=True), oof_aligned.reset_index(drop=True), report


def validate_targets_against_oof(
    *,
    merged_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    strict: bool,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    for target in EMOTION_COLS + VAD_COLS:
        if target not in merged_df.columns:
            raise ValueError(f"Merged CSV is missing target column '{target}'.")
        oof_true_col = f"true_{target}"
        if oof_true_col not in oof_df.columns:
            raise ValueError(f"OOF CSV is missing required column '{oof_true_col}'.")

        x = pd.to_numeric(merged_df[target], errors="coerce")
        y = pd.to_numeric(oof_df[oof_true_col], errors="coerce")
        mismatch = (x.isna() ^ y.isna()) | ((x - y).abs() > 1e-9)
        mismatch_count = int(mismatch.sum())
        checks[target] = {
            "mismatch_count": mismatch_count,
            "matches": bool(mismatch_count == 0),
        }
        if mismatch_count > 0:
            _raise_or_warn(strict, f"Target/OFF truth mismatch for '{target}' (count={mismatch_count}).")
    return checks


def add_rank_within_target(df: pd.DataFrame, rank_key: str) -> pd.DataFrame:
    out = df.copy()
    out["rank_within_target"] = 0
    sort_tiebreak = rank_key if rank_key in out.columns else out.columns[0]
    for target, grp in out.groupby("target", sort=False):
        sortable = grp.copy()
        sortable["_score"] = pd.to_numeric(sortable["importance_mean"], errors="coerce").fillna(-np.inf)
        order = sortable.sort_values(["_score", sort_tiebreak], ascending=[False, True], kind="mergesort").index
        for i, idx in enumerate(order, start=1):
            out.loc[idx, "rank_within_target"] = i
    out["rank_within_target"] = out["rank_within_target"].astype(int)
    return out


def build_emotion_flags(
    *,
    diag_df: pd.DataFrame,
    y_em_df: pd.DataFrame,
    max_folds: int,
    rare_pos_threshold: int,
) -> pd.DataFrame:
    diag_map = {}
    if "emotion" in diag_df.columns:
        for _, row in diag_df.iterrows():
            diag_map[str(row["emotion"])] = row.to_dict()

    rows: List[Dict[str, Any]] = []
    n = int(len(y_em_df))
    for emo in EMOTION_COLS:
        y = pd.to_numeric(y_em_df[emo], errors="coerce").fillna(0).astype(int).to_numpy()
        default_pos = int(y.sum())
        default_neg = int(n - default_pos)
        d = diag_map.get(emo, {})
        n_pos = _safe_int(d.get("n_pos", default_pos), default_pos)
        n_neg = _safe_int(d.get("n_neg", default_neg), default_neg)
        status = str(d.get("status", "trained" if n_pos > 0 and n_neg > 0 else "degenerate_constant"))
        n_splits = _safe_int(d.get("n_splits", min(max_folds, n_pos, n_neg) if n_pos > 0 and n_neg > 0 else 0), 0)
        is_rare = bool(n_pos < rare_pos_threshold)

        if status in {"degenerate_constant", "insufficient_minority"}:
            interpretation_level = "not_estimable"
            exclude_headline = True
            reason = status
        elif is_rare:
            interpretation_level = "exploratory_rare"
            exclude_headline = True
            reason = "rare_positive_count"
        else:
            interpretation_level = "stable"
            exclude_headline = False
            reason = "eligible"

        rows.append(
            {
                "target": emo,
                "target_type": "emotion",
                "status": status,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_splits": n_splits,
                "is_rare": bool(is_rare),
                "interpretation_level": interpretation_level,
                "exclude_headline": bool(exclude_headline),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_vad_flags(*, n_rows: int, max_folds: int) -> pd.DataFrame:
    n_splits = int(min(max_folds, n_rows))
    status = "trained" if n_splits >= 2 else "insufficient_rows"
    interpretation_level = "stable" if status == "trained" else "not_estimable"
    exclude_headline = status != "trained"

    rows: List[Dict[str, Any]] = []
    for dim in VAD_COLS:
        rows.append(
            {
                "target": dim,
                "target_type": "vad",
                "status": status,
                "n_pos": np.nan,
                "n_neg": np.nan,
                "n_splits": n_splits,
                "is_rare": False,
                "interpretation_level": interpretation_level,
                "exclude_headline": bool(exclude_headline),
                "reason": "eligible" if status == "trained" else "insufficient_rows",
            }
        )
    return pd.DataFrame(rows)


def compute_emotion_importance(
    *,
    X: np.ndarray,
    y_em_df: pd.DataFrame,
    feature_cols: List[str],
    feature_to_family: Dict[str, str],
    family_to_indices: Dict[str, List[int]],
    flags_df: pd.DataFrame,
    n_repeats: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_records: List[Dict[str, Any]] = []
    family_records: List[Dict[str, Any]] = []

    for target_idx, target in enumerate(EMOTION_COLS):
        flag_row = flags_df.loc[flags_df["target"] == target].iloc[0].to_dict()
        status = str(flag_row["status"])
        interpretation_level = str(flag_row["interpretation_level"])
        n_splits = int(flag_row["n_splits"])

        if status != "trained" or n_splits < 2:
            for feat in feature_cols:
                feature_records.append(
                    {
                        "target": target,
                        "feature": feat,
                        "family": feature_to_family.get(feat, "Other"),
                        "metric_name": "average_precision",
                        "importance_mean": np.nan,
                        "importance_std": np.nan,
                        "importance_median": np.nan,
                        "baseline_metric_mean": np.nan,
                        "n_folds_used": 0,
                        "n_repeats": int(n_repeats),
                        "status": status,
                        "interpretation_level": interpretation_level,
                    }
                )
            for family in family_to_indices:
                family_records.append(
                    {
                        "target": target,
                        "family": family,
                        "metric_name": "average_precision",
                        "importance_mean": np.nan,
                        "importance_std": np.nan,
                        "importance_median": np.nan,
                        "baseline_metric_mean": np.nan,
                        "n_folds_used": 0,
                        "n_repeats": int(n_repeats),
                        "status": status,
                        "interpretation_level": interpretation_level,
                    }
                )
            continue

        y = pd.to_numeric(y_em_df[target], errors="coerce").fillna(0).astype(int).to_numpy()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        feat_deltas: Dict[str, List[float]] = {f: [] for f in feature_cols}
        fam_deltas: Dict[str, List[float]] = {fam: [] for fam in family_to_indices}
        baseline_metrics: List[float] = []
        valid_folds = 0

        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            model = build_emotion_model()
            model.fit(X[tr_idx], y[tr_idx])
            X_te = X[te_idx]
            y_te = y[te_idx]

            base_prob = model.predict_proba(X_te)[:, 1]
            baseline_ap = _safe_average_precision(y_te, base_prob)
            if not np.isfinite(baseline_ap):
                continue
            baseline_metrics.append(baseline_ap)
            valid_folds += 1

            # Feature-level permutations
            for feat_idx, feat in enumerate(feature_cols):
                x_work = X_te.copy()
                original_col = x_work[:, feat_idx].copy()
                for rep in range(n_repeats):
                    rng = np.random.default_rng(seed + target_idx * 1_000_000 + fold_idx * 10_000 + feat_idx * 101 + rep)
                    perm = rng.permutation(x_work.shape[0])
                    x_work[:, feat_idx] = original_col[perm]
                    perm_prob = model.predict_proba(x_work)[:, 1]
                    perm_ap = _safe_average_precision(y_te, perm_prob)
                    if np.isfinite(perm_ap):
                        feat_deltas[feat].append(float(baseline_ap - perm_ap))
                x_work[:, feat_idx] = original_col

            # Family-level grouped permutations
            for fam_idx, (family, indices) in enumerate(family_to_indices.items()):
                x_work = X_te.copy()
                original_block = x_work[:, indices].copy()
                for rep in range(n_repeats):
                    rng = np.random.default_rng(
                        seed + target_idx * 1_000_000 + fold_idx * 10_000 + fam_idx * 313 + rep
                    )
                    perm = rng.permutation(x_work.shape[0])
                    x_work[:, indices] = original_block[perm, :]
                    perm_prob = model.predict_proba(x_work)[:, 1]
                    perm_ap = _safe_average_precision(y_te, perm_prob)
                    if np.isfinite(perm_ap):
                        fam_deltas[family].append(float(baseline_ap - perm_ap))
                x_work[:, indices] = original_block

        baseline_mean = _metric_mean(baseline_metrics)
        for feat in feature_cols:
            row = {
                "target": target,
                "feature": feat,
                "family": feature_to_family.get(feat, "Other"),
                "metric_name": "average_precision",
                "baseline_metric_mean": baseline_mean,
                "n_folds_used": int(valid_folds),
                "n_repeats": int(n_repeats),
                "status": status,
                "interpretation_level": interpretation_level,
            }
            row.update(_delta_stats(feat_deltas[feat]))
            feature_records.append(row)

        for family in family_to_indices:
            row = {
                "target": target,
                "family": family,
                "metric_name": "average_precision",
                "baseline_metric_mean": baseline_mean,
                "n_folds_used": int(valid_folds),
                "n_repeats": int(n_repeats),
                "status": status,
                "interpretation_level": interpretation_level,
            }
            row.update(_delta_stats(fam_deltas[family]))
            family_records.append(row)

    feature_df = pd.DataFrame(feature_records)
    family_df = pd.DataFrame(family_records)
    return feature_df, family_df


def compute_vad_importance(
    *,
    X: np.ndarray,
    y_vad_df: pd.DataFrame,
    feature_cols: List[str],
    feature_to_family: Dict[str, str],
    family_to_indices: Dict[str, List[int]],
    flags_df: pd.DataFrame,
    n_repeats: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_records: List[Dict[str, Any]] = []
    family_records: List[Dict[str, Any]] = []

    flag_map = {r["target"]: r for _, r in flags_df.iterrows()}
    n_rows = X.shape[0]
    n_splits = int(flags_df["n_splits"].iloc[0]) if not flags_df.empty else 0
    status = str(flags_df["status"].iloc[0]) if not flags_df.empty else "insufficient_rows"

    if status != "trained" or n_splits < 2:
        for dim in VAD_COLS:
            interpretation_level = str(flag_map.get(dim, {}).get("interpretation_level", "not_estimable"))
            for feat in feature_cols:
                feature_records.append(
                    {
                        "target": dim,
                        "feature": feat,
                        "family": feature_to_family.get(feat, "Other"),
                        "metric_name": "mae",
                        "importance_mean": np.nan,
                        "importance_std": np.nan,
                        "importance_median": np.nan,
                        "baseline_metric_mean": np.nan,
                        "n_folds_used": 0,
                        "n_repeats": int(n_repeats),
                        "status": status,
                        "interpretation_level": interpretation_level,
                    }
                )
            for family in family_to_indices:
                family_records.append(
                    {
                        "target": dim,
                        "family": family,
                        "metric_name": "mae",
                        "importance_mean": np.nan,
                        "importance_std": np.nan,
                        "importance_median": np.nan,
                        "baseline_metric_mean": np.nan,
                        "n_folds_used": 0,
                        "n_repeats": int(n_repeats),
                        "status": status,
                        "interpretation_level": interpretation_level,
                    }
                )
        return pd.DataFrame(feature_records), pd.DataFrame(family_records)

    y_vad = y_vad_df[VAD_COLS].to_numpy(dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    feat_deltas: Dict[str, Dict[str, List[float]]] = {dim: {f: [] for f in feature_cols} for dim in VAD_COLS}
    fam_deltas: Dict[str, Dict[str, List[float]]] = {
        dim: {fam: [] for fam in family_to_indices} for dim in VAD_COLS
    }
    baseline_metrics: Dict[str, List[float]] = {dim: [] for dim in VAD_COLS}
    valid_folds = 0

    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(np.arange(n_rows))):
        model = build_vad_model()
        model.fit(X[tr_idx], y_vad[tr_idx])
        x_te = X[te_idx]
        y_te = y_vad[te_idx]

        base_pred = model.predict(x_te)
        base_mae = [float(mean_absolute_error(y_te[:, i], base_pred[:, i])) for i in range(len(VAD_COLS))]
        valid_folds += 1
        for i, dim in enumerate(VAD_COLS):
            baseline_metrics[dim].append(base_mae[i])

        for feat_idx, feat in enumerate(feature_cols):
            x_work = x_te.copy()
            original_col = x_work[:, feat_idx].copy()
            for rep in range(n_repeats):
                rng = np.random.default_rng(seed + fold_idx * 10_000 + feat_idx * 101 + rep)
                perm = rng.permutation(x_work.shape[0])
                x_work[:, feat_idx] = original_col[perm]
                perm_pred = model.predict(x_work)
                for i, dim in enumerate(VAD_COLS):
                    perm_mae = float(mean_absolute_error(y_te[:, i], perm_pred[:, i]))
                    feat_deltas[dim][feat].append(float(perm_mae - base_mae[i]))
            x_work[:, feat_idx] = original_col

        for fam_idx, (family, indices) in enumerate(family_to_indices.items()):
            x_work = x_te.copy()
            original_block = x_work[:, indices].copy()
            for rep in range(n_repeats):
                rng = np.random.default_rng(seed + fold_idx * 10_000 + fam_idx * 313 + rep)
                perm = rng.permutation(x_work.shape[0])
                x_work[:, indices] = original_block[perm, :]
                perm_pred = model.predict(x_work)
                for i, dim in enumerate(VAD_COLS):
                    perm_mae = float(mean_absolute_error(y_te[:, i], perm_pred[:, i]))
                    fam_deltas[dim][family].append(float(perm_mae - base_mae[i]))
            x_work[:, indices] = original_block

    for dim in VAD_COLS:
        interpretation_level = str(flag_map.get(dim, {}).get("interpretation_level", "stable"))
        status_dim = str(flag_map.get(dim, {}).get("status", status))
        baseline_mean = _metric_mean(baseline_metrics[dim])
        for feat in feature_cols:
            row = {
                "target": dim,
                "feature": feat,
                "family": feature_to_family.get(feat, "Other"),
                "metric_name": "mae",
                "baseline_metric_mean": baseline_mean,
                "n_folds_used": int(valid_folds),
                "n_repeats": int(n_repeats),
                "status": status_dim,
                "interpretation_level": interpretation_level,
            }
            row.update(_delta_stats(feat_deltas[dim][feat]))
            feature_records.append(row)

        for family in family_to_indices:
            row = {
                "target": dim,
                "family": family,
                "metric_name": "mae",
                "baseline_metric_mean": baseline_mean,
                "n_folds_used": int(valid_folds),
                "n_repeats": int(n_repeats),
                "status": status_dim,
                "interpretation_level": interpretation_level,
            }
            row.update(_delta_stats(fam_deltas[dim][family]))
            family_records.append(row)

    return pd.DataFrame(feature_records), pd.DataFrame(family_records)


def top_importances(
    *,
    df: pd.DataFrame,
    item_col: str,
    eligible_targets: Iterable[str],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    subset = df[df["target"].isin(list(eligible_targets))].copy()
    if subset.empty:
        return []
    subset["importance_pos"] = pd.to_numeric(subset["importance_mean"], errors="coerce").fillna(0.0).clip(lower=0.0)
    agg = (
        subset.groupby(item_col, as_index=False)
        .agg(mean_positive_importance=("importance_pos", "mean"), n_targets=("target", "nunique"))
        .sort_values(["mean_positive_importance", item_col], ascending=[False, True], kind="mergesort")
    )
    return agg.head(top_k).to_dict(orient="records")


def run_stage8_analysis(
    *,
    merged_csv: Path,
    stage7_metrics_json: Path,
    stage7_label_diagnostics_csv: Path,
    stage7_oof_csv: Path,
    out_dir: Path,
    n_repeats: int = 5,
    seed: int | None = None,
    max_folds: int | None = None,
    rare_pos_threshold: int = 10,
    strict: bool = True,
) -> Dict[str, Any]:
    if not merged_csv.exists():
        raise FileNotFoundError(f"Merged CSV not found: {merged_csv}")
    if not stage7_metrics_json.exists():
        raise FileNotFoundError(f"Stage 7 metrics JSON not found: {stage7_metrics_json}")
    if not stage7_label_diagnostics_csv.exists():
        raise FileNotFoundError(f"Stage 7 label diagnostics CSV not found: {stage7_label_diagnostics_csv}")
    if not stage7_oof_csv.exists():
        raise FileNotFoundError(f"Stage 7 OOF CSV not found: {stage7_oof_csv}")
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_df = pd.read_csv(merged_csv)
    oof_df = pd.read_csv(stage7_oof_csv)
    with stage7_metrics_json.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    diag_df = pd.read_csv(stage7_label_diagnostics_csv)

    config = metrics.get("config", {})
    seed = int(seed if seed is not None else config.get("seed", 42))
    max_folds = int(max_folds if max_folds is not None else config.get("max_folds", 5))

    feature_cols = list(metrics.get("feature_columns", []))
    if not feature_cols:
        raise ValueError("Stage 7 metrics JSON does not provide feature_columns.")
    missing_features = [c for c in feature_cols if c not in merged_df.columns]
    if missing_features:
        raise ValueError(f"Merged CSV missing {len(missing_features)} canonical feature columns.")

    merged_aligned, oof_aligned, alignment_report = validate_and_align_inputs(
        merged_df=merged_df, oof_df=oof_df, strict=strict
    )
    truth_checks = validate_targets_against_oof(merged_df=merged_aligned, oof_df=oof_aligned, strict=strict)
    alignment_report["target_truth_checks"] = truth_checks

    x_df = merged_aligned[feature_cols].copy()
    for col in feature_cols:
        x_df[col] = pd.to_numeric(x_df[col], errors="coerce")
    X = x_df.to_numpy(dtype=float)

    y_em_df = merged_aligned[EMOTION_COLS].copy()
    y_vad_df = merged_aligned[VAD_COLS].copy()
    for col in EMOTION_COLS + VAD_COLS:
        merged_aligned[col] = pd.to_numeric(merged_aligned[col], errors="coerce")
    if merged_aligned[EMOTION_COLS + VAD_COLS].isna().any().any():
        raise ValueError("Merged targets contain NaN after numeric coercion.")

    feature_to_family, family_to_features, family_to_indices = build_family_index(feature_cols)
    family_presence = {fam: len(family_to_features.get(fam, [])) for fam in REQUIRED_FAMILIES}
    alignment_report["family_presence"] = family_presence

    emotion_flags = build_emotion_flags(
        diag_df=diag_df,
        y_em_df=y_em_df,
        max_folds=max_folds,
        rare_pos_threshold=rare_pos_threshold,
    )
    vad_flags = build_vad_flags(n_rows=len(merged_aligned), max_folds=max_folds)
    flags_df = pd.concat([emotion_flags, vad_flags], ignore_index=True)

    em_feature_df, em_family_df = compute_emotion_importance(
        X=X,
        y_em_df=y_em_df,
        feature_cols=feature_cols,
        feature_to_family=feature_to_family,
        family_to_indices=family_to_indices,
        flags_df=emotion_flags,
        n_repeats=n_repeats,
        seed=seed,
    )
    vad_feature_df, vad_family_df = compute_vad_importance(
        X=X,
        y_vad_df=y_vad_df,
        feature_cols=feature_cols,
        feature_to_family=feature_to_family,
        family_to_indices=family_to_indices,
        flags_df=vad_flags,
        n_repeats=n_repeats,
        seed=seed,
    )

    em_feature_df = add_rank_within_target(em_feature_df, rank_key="feature")
    em_family_df = add_rank_within_target(em_family_df, rank_key="family")
    vad_feature_df = add_rank_within_target(vad_feature_df, rank_key="feature")
    vad_family_df = add_rank_within_target(vad_family_df, rank_key="family")

    alignment_out = out_dir / "stage8_input_alignment_report.json"
    em_feature_out = out_dir / "stage8_emotion_feature_importance.csv"
    em_family_out = out_dir / "stage8_emotion_family_importance.csv"
    vad_feature_out = out_dir / "stage8_vad_feature_importance.csv"
    vad_family_out = out_dir / "stage8_vad_family_importance.csv"
    flags_out = out_dir / "stage8_target_interpretation_flags.csv"
    summary_out = out_dir / "stage8_importance_summary_report.json"

    alignment_out.write_text(json.dumps(alignment_report, indent=2), encoding="utf-8")
    em_feature_df.to_csv(em_feature_out, index=False)
    em_family_df.to_csv(em_family_out, index=False)
    vad_feature_df.to_csv(vad_feature_out, index=False)
    vad_family_df.to_csv(vad_family_out, index=False)
    flags_df.to_csv(flags_out, index=False)

    eligible_emotion_targets = flags_df[
        (flags_df["target_type"] == "emotion") & (~flags_df["exclude_headline"])
    ]["target"].tolist()
    eligible_vad_targets = flags_df[
        (flags_df["target_type"] == "vad") & (~flags_df["exclude_headline"])
    ]["target"].tolist()

    summary: Dict[str, Any] = {
        "inputs": {
            "merged_csv": str(merged_csv),
            "stage7_metrics_json": str(stage7_metrics_json),
            "stage7_label_diagnostics_csv": str(stage7_label_diagnostics_csv),
            "stage7_oof_csv": str(stage7_oof_csv),
        },
        "config": {
            "n_repeats": int(n_repeats),
            "seed": int(seed),
            "max_folds": int(max_folds),
            "rare_pos_threshold": int(rare_pos_threshold),
            "strict": bool(strict),
        },
        "integrity": alignment_report,
        "target_status_overview": flags_df.to_dict(orient="records"),
        "headline": {
            "emotion_feature_top_nonrare": top_importances(
                df=em_feature_df,
                item_col="feature",
                eligible_targets=eligible_emotion_targets,
                top_k=10,
            ),
            "emotion_family_top_nonrare": top_importances(
                df=em_family_df,
                item_col="family",
                eligible_targets=eligible_emotion_targets,
                top_k=10,
            ),
            "vad_feature_top": top_importances(
                df=vad_feature_df,
                item_col="feature",
                eligible_targets=eligible_vad_targets,
                top_k=10,
            ),
            "vad_family_top": top_importances(
                df=vad_family_df,
                item_col="family",
                eligible_targets=eligible_vad_targets,
                top_k=10,
            ),
        },
        "caveats": [
            "Permutation importances are fold-wise held-out estimates; they are not causal effects.",
            "Degenerate targets are not estimable and are excluded from headline claims.",
            "Rare emotion targets are marked exploratory_rare and excluded from headline claims.",
        ],
        "artifacts": {
            "alignment_report_json": str(alignment_out),
            "emotion_feature_importance_csv": str(em_feature_out),
            "emotion_family_importance_csv": str(em_family_out),
            "vad_feature_importance_csv": str(vad_feature_out),
            "vad_family_importance_csv": str(vad_family_out),
            "target_interpretation_flags_csv": str(flags_out),
            "summary_report_json": str(summary_out),
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Stage 8A feature importance complete")
    print(f"- Matched rows: {alignment_report['matched_count']}")
    print(f"- Emotion feature importance: {em_feature_out}")
    print(f"- Emotion family importance: {em_family_out}")
    print(f"- VAD feature importance: {vad_feature_out}")
    print(f"- VAD family importance: {vad_family_out}")
    print(f"- Summary report: {summary_out}")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage 8A validation-safe feature importance analysis.")
    ap.add_argument(
        "--merged-csv",
        type=Path,
        default=outputs_path("analysis_out", "gemini_unified_merged.csv"),
    )
    ap.add_argument(
        "--stage7-metrics-json",
        type=Path,
        default=outputs_path("training", "stage7_gemini784_metrics.json"),
    )
    ap.add_argument(
        "--stage7-label-diagnostics-csv",
        type=Path,
        default=outputs_path("training", "stage7_gemini784_label_diagnostics.csv"),
    )
    ap.add_argument(
        "--stage7-oof-csv",
        type=Path,
        default=outputs_path("training", "stage7_gemini784_oof_predictions.csv"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=outputs_path("analysis_out"),
    )
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--rare-pos-threshold", type=int, default=10)
    ap.add_argument("--strict", dest="strict", action="store_true", default=True)
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_stage8_analysis(
        merged_csv=args.merged_csv,
        stage7_metrics_json=args.stage7_metrics_json,
        stage7_label_diagnostics_csv=args.stage7_label_diagnostics_csv,
        stage7_oof_csv=args.stage7_oof_csv,
        out_dir=args.out_dir,
        n_repeats=args.n_repeats,
        seed=args.seed,
        max_folds=args.max_folds,
        rare_pos_threshold=args.rare_pos_threshold,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
