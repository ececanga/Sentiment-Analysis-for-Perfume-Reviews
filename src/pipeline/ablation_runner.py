#!/usr/bin/env python3
"""
Ablation runner for: data/perfume_level/Unified_Emotion_Profile_Perfume.csv (Stage 5B)

Writes all outputs under:
  outputs/ablation_results/

Change vs previous version:
- When --write_matrices is enabled, per-variant outputs are named directly as:
    distance_matrix_<VARIANT>.csv
    nearest_neighbors_<VARIANT>.csv
  inside each variant folder.

Run:
  python "src/pipeline/ablation_runner.py"

With matrices:
  python "src/pipeline/ablation_runner.py" --write_matrices
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.paths import data_path, outputs_path
# Optional sklearn
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def zscore_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd

def standardize(X: np.ndarray) -> np.ndarray:
    if SKLEARN_OK:
        return StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    return zscore_np(X)

def cosine_distance_matrix(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Returns NxN cosine distance matrix (1 - cosine similarity).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, eps, norms)
    Xn = X / norms
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim

def upper_triangle_values(M: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(M, k=1)
    return M[idx]

def rankdata(a: np.ndarray) -> np.ndarray:
    """
    Average ranks for ties. Deterministic.
    """
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)

    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0
            ranks[order[i:j+1]] = avg
        i = j + 1
    return ranks

def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 3:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom == 0:
        return float("nan")
    return float((rx @ ry) / denom)

def knn_indices_from_distance(D: np.ndarray, k: int) -> np.ndarray:
    """
    For each row, returns indices of k nearest neighbors (excluding self).
    """
    idx = np.argsort(D, axis=1)
    return idx[:, 1:k+1]

def knn_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean overlap ratio across rows: |intersection| / k
    """
    n, k = a.shape
    overlaps = []
    for i in range(n):
        overlaps.append(len(set(a[i]).intersection(set(b[i]))) / k)
    return float(np.mean(overlaps))

def safe_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.fillna(0.0)

def kmeans_labels(X: np.ndarray, k: int, seed: int = 42) -> Optional[np.ndarray]:
    if not SKLEARN_OK:
        return None
    if X.shape[0] < k:
        return None
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    return km.fit_predict(X)


# ----------------------------
# Column grouping (schema)
# ----------------------------

def build_column_groups(columns: List[str]) -> Dict[str, List[str]]:
    nrc = sorted([c for c in columns if c.startswith("NRC_")])
    vad = sorted([c for c in columns if c.startswith("VAD_")])
    hart = sorted([c for c in columns if c.startswith("Hartmann_")])
    sav = sorted([c for c in columns if c.startswith("Savani_")])
    goe = sorted([c for c in columns if c.startswith("GoEmotions_")])
    car = sorted([c for c in columns if c.startswith("Cardiff_")])

    labels = sorted({c for c in columns if c.endswith("_Label")})

    transformer_scores = sorted(hart + sav + goe + car)
    lexicon_scores = sorted(nrc + vad)

    return {
        "NRC": nrc,
        "VAD": vad,
        "Hartmann": hart,
        "Savani": sav,
        "GoEmotions": goe,
        "Cardiff": car,
        "TransformerScores": transformer_scores,
        "LexiconScores": lexicon_scores,
        "Labels": labels,
    }

def build_variants(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    full = groups["LexiconScores"] + groups["TransformerScores"]

    variants = {
        "FULL_HYBRID": full,
        "LEXICON_ONLY": groups["LexiconScores"],
        "TRANSFORMER_ONLY": groups["TransformerScores"],
        "NO_NRC": [c for c in full if not c.startswith("NRC_")],
        "NO_VAD": [c for c in full if not c.startswith("VAD_")],
        "HARTMANN_ONLY": groups["Hartmann"],
        "SAVANI_ONLY": groups["Savani"],
        "GOEMOTIONS_ONLY": groups["GoEmotions"],
        "CARDIFF_ONLY": groups["Cardiff"],
        "LABELS_ONLY": groups["Labels"],  # handled separately
    }

    return {k: v for k, v in variants.items() if len(v) > 0}


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=data_path("perfume_level", "Unified_Emotion_Profile_Perfume.csv"),
        help="Input unified perfume profile CSV (default: data/perfume_level/Unified_Emotion_Profile_Perfume.csv)",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=outputs_path("ablation_results"),
        help="Output directory (default: outputs/ablation_results)",
    )
    ap.add_argument("--knn_k", type=int, default=10, help="k for nearest-neighbor overlap")
    ap.add_argument("--clusters_k", type=int, default=8, help="k for KMeans clustering (if sklearn available)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--write_matrices",
        action="store_true",
        help="Write distance matrices + neighbor lists per variant (bigger outputs).",
    )
    args = ap.parse_args()

    in_path = args.input
    outdir = args.outdir
    ensure_dir(outdir)

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    if "Perfume" not in df.columns:
        raise ValueError("Expected a 'Perfume' column in the input CSV.")

    # Ensure unique perfumes
    df = df.drop_duplicates(subset=["Perfume"]).reset_index(drop=True)
    perfume_names = df["Perfume"].astype(str).tolist()

    groups = build_column_groups(list(df.columns))
    variants = build_variants(groups)

    if "FULL_HYBRID" not in variants:
        raise ValueError("Could not form FULL_HYBRID (no numeric feature columns found).")

    # ---- Baseline distances / neighbors ----
    base_cols = variants["FULL_HYBRID"]
    X_base = safe_numeric_df(df, base_cols).to_numpy(dtype=float)
    X_base = standardize(X_base)
    D_base = cosine_distance_matrix(X_base)
    base_vec = upper_triangle_values(D_base)

    k_eff = min(args.knn_k, max(1, len(perfume_names) - 1))
    base_knn = knn_indices_from_distance(D_base, k=k_eff)

    base_cluster_labels = kmeans_labels(X_base, k=args.clusters_k, seed=args.seed)

    # ---- Ranking axes (simple, interpretable) ----
    ranking_axes = {}
    if "VAD_valence" in df.columns:
        ranking_axes["RANK_VAD_VALENCE"] = ("VAD_valence",)
    if "NRC_positive" in df.columns and "NRC_negative" in df.columns:
        ranking_axes["RANK_NRC_POSNEG"] = ("NRC_positive", "NRC_negative")
    if "Hartmann_joy" in df.columns:
        ranking_axes["RANK_HARTMANN_JOY"] = ("Hartmann_joy",)
    if "GoEmotions_joy" in df.columns:
        ranking_axes["RANK_GOEMOTIONS_JOY"] = ("GoEmotions_joy",)

    baseline_ranks = {}
    for key, cols_ in ranking_axes.items():
        if len(cols_) == 1:
            s = pd.to_numeric(df[cols_[0]], errors="coerce").fillna(0.0).to_numpy()
        else:
            pos = pd.to_numeric(df[cols_[0]], errors="coerce").fillna(0.0).to_numpy()
            neg = pd.to_numeric(df[cols_[1]], errors="coerce").fillna(0.0).to_numpy()
            s = pos - neg
        baseline_ranks[key] = rankdata(s)

    summary_rows = []

    # ---- Iterate variants ----
    for vname, vcols in variants.items():
        v_out = outdir / vname
        ensure_dir(v_out)

        # LABELS_ONLY: categorical; report simple diagnostics only
        if vname == "LABELS_ONLY":
            label_cols = [c for c in vcols if c in df.columns]
            row = {
                "variant": vname,
                "n_perfumes": len(perfume_names),
                "n_features": len(label_cols),
            }
            for c in label_cols:
                row[f"uniq_{c}"] = int(df[c].astype(str).nunique())
            summary_rows.append(row)
            continue

        # Numeric variant
        X = safe_numeric_df(df, vcols).to_numpy(dtype=float)
        X = standardize(X)
        D = cosine_distance_matrix(X)
        vec = upper_triangle_values(D)

        # 1) Pairwise geometry preservation vs baseline
        spearman_dist = spearman_corr(base_vec, vec)

        # 2) kNN overlap
        knn = knn_indices_from_distance(D, k=k_eff)
        knn_ov = knn_overlap(base_knn, knn)

        # 3) Cluster agreement vs baseline
        ari = float("nan")
        if base_cluster_labels is not None:
            v_cluster_labels = kmeans_labels(X, k=args.clusters_k, seed=args.seed)
            if v_cluster_labels is not None:
                ari = float(adjusted_rand_score(base_cluster_labels, v_cluster_labels))

        # 4) Ranking stability on a few axes
        rank_scores = {}
        for key, cols_ in ranking_axes.items():
            if len(cols_) == 1 and cols_[0] not in vcols:
                continue

            if len(cols_) == 1:
                s = pd.to_numeric(df[cols_[0]], errors="coerce").fillna(0.0).to_numpy()
            else:
                pos = pd.to_numeric(df[cols_[0]], errors="coerce").fillna(0.0).to_numpy()
                neg = pd.to_numeric(df[cols_[1]], errors="coerce").fillna(0.0).to_numpy()
                s = pos - neg

            vr = rankdata(s)
            br = baseline_ranks[key]
            rank_scores[key] = spearman_corr(br, vr)

        row = {
            "variant": vname,
            "n_perfumes": len(perfume_names),
            "n_features": len(vcols),
            "spearman_pairwise_cosdist_vs_full": spearman_dist,
            "knn_overlap_vs_full": knn_ov,
            "kmeans_ari_vs_full": ari,
        }
        row.update(rank_scores)
        summary_rows.append(row)

        # Optional: write matrices / neighbor lists with DIRECT NAMES
        if args.write_matrices:
            dist_path = v_out / f"distance_matrix_{vname}.csv"
            nn_path = v_out / f"nearest_neighbors_{vname}.csv"

            pd.DataFrame(D, index=perfume_names, columns=perfume_names).to_csv(dist_path)

            nn_rows = []
            for i, p in enumerate(perfume_names):
                for j in knn[i]:
                    nn_rows.append({
                        "Perfume": p,
                        "Neighbor": perfume_names[j],
                        "cosine_distance": float(D[i, j]),
                    })
            pd.DataFrame(nn_rows).to_csv(nn_path, index=False)

    summary = pd.DataFrame(summary_rows).sort_values("variant").reset_index(drop=True)
    summary_path = outdir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"✅ Ablation done. Summary written to: {summary_path}")
    if args.write_matrices:
        print(f"✅ Wrote per-variant matrices with direct names: distance_matrix_<VARIANT>.csv and nearest_neighbors_<VARIANT>.csv")
    if not SKLEARN_OK:
        print("ℹ️ scikit-learn not found; KMeans/ARI metrics skipped. Install with: pip install scikit-learn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
