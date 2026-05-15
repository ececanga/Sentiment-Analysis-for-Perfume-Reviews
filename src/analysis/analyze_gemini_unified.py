# Date Created: 2026-04-13 23:02:44+03:00
# Date Last Modified: 2026-05-15

"""Minimal table-first analysis for Unified profile + Gemini labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.labeling.strict_schema import normalize_perfume_id, perfume_name_hash10, perfume_name_hash4
from src.utils.paths import data_path, labels_path, outputs_path

EMOTIONS = ["joy", "sadness", "anger", "fear", "trust", "surprise", "anticipation", "disgust"]
PRIMARY_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise"]
VAD_DIMS = ["valence", "arousal", "dominance"]
MODEL_LABEL_COLS = ["JHartmann_Label", "Savani_Label", "GoEmotions_Label", "Cardiff_Label"]

COARSE_LABEL_MAP = {
    "joy": "joy",
    "love": "joy",
    "optimism": "joy",
    "admiration": "joy",
    "approval": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "relief": "joy",
    "pride": "joy",
    "desire": "joy",
    "trust": "joy",
    "sadness": "sadness",
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "anger",
    "fear": "fear",
    "nervousness": "fear",
    "anxiety": "fear",
    "embarrassment": "fear",
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "neutral": "neutral",
}


def _ensure_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def build_ids_from_names(names: Iterable[str]) -> pd.DataFrame:
    rows = []
    for raw_name in names:
        name = str(raw_name or "").strip()
        if not name:
            rows.append({"perfume_id": "", "perfume_id_hash": ""})
            continue
        rows.append(
            {
                "perfume_name": name,
                "base_slug": normalize_perfume_id(name),
                "perfume_id_hash": perfume_name_hash10(name),
                "hash4": perfume_name_hash4(name),
            }
        )

    tmp = pd.DataFrame(rows)
    if tmp.empty:
        return pd.DataFrame(columns=["perfume_id", "perfume_id_hash"])

    tmp["perfume_id"] = ""
    used_ids: set[str] = set()
    for slug, grp in tmp.groupby("base_slug", sort=False):
        idxs = grp.index.tolist()
        if len(idxs) == 1:
            candidate = slug
            if candidate in used_ids:
                candidate = f"{slug}-{tmp.loc[idxs[0], 'hash4']}"
            tmp.loc[idxs[0], "perfume_id"] = candidate
            used_ids.add(candidate)
            continue

        for i in idxs:
            candidate = f"{slug}-{tmp.loc[i, 'hash4']}"
            if candidate in used_ids:
                candidate = f"{slug}-{tmp.loc[i, 'perfume_id_hash'][:6]}"
            tmp.loc[i, "perfume_id"] = candidate
            used_ids.add(candidate)

    return tmp[["perfume_id", "perfume_id_hash"]]


def prepare_unified(unified_df: pd.DataFrame, perfume_col: str) -> pd.DataFrame:
    if perfume_col not in unified_df.columns:
        raise ValueError(f"Unified file is missing perfume column '{perfume_col}'.")
    out = unified_df.copy()
    out["perfume_name_unified"] = _ensure_str_series(out[perfume_col])
    ids = build_ids_from_names(out["perfume_name_unified"])
    out["perfume_id"] = ids["perfume_id"]
    out["perfume_id_hash_unified"] = ids["perfume_id_hash"]
    return out


def prepare_gemini(gemini_df: pd.DataFrame) -> pd.DataFrame:
    out = gemini_df.copy()

    if "perfume_name" in out.columns:
        out["perfume_name_gemini"] = _ensure_str_series(out["perfume_name"])
    else:
        out["perfume_name_gemini"] = ""

    if "perfume_id" in out.columns:
        out["perfume_id"] = _ensure_str_series(out["perfume_id"])
    else:
        if "perfume_name_gemini" not in out.columns:
            raise ValueError("Gemini file must include either perfume_id or perfume_name.")
        ids = build_ids_from_names(out["perfume_name_gemini"])
        out["perfume_id"] = ids["perfume_id"]
        out["perfume_id_hash"] = ids["perfume_id_hash"]

    if "perfume_id_hash" in out.columns:
        out["perfume_id_hash_gemini"] = _ensure_str_series(out["perfume_id_hash"])
    else:
        hash_df = build_ids_from_names(out["perfume_name_gemini"])
        out["perfume_id_hash_gemini"] = hash_df["perfume_id_hash"]

    for col in EMOTIONS + VAD_DIMS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _duplicate_ids(df: pd.DataFrame, id_col: str) -> List[str]:
    return sorted(df.loc[df[id_col].duplicated(keep=False), id_col].dropna().astype(str).unique().tolist())


def _coarsen_label(label: Any) -> str:
    if not isinstance(label, str) or not label.strip():
        return "neutral"
    key = label.strip().lower()
    return COARSE_LABEL_MAP.get(key, "neutral")


def _gemini_primary_label(row: pd.Series) -> str:
    active = [emo for emo in PRIMARY_EMOTIONS if float(row.get(emo, 0) or 0) >= 0.5]
    if len(active) == 0:
        return "neutral"
    if len(active) == 1:
        return active[0]
    return "multi"


def binary_nrc_agreement(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for emo in EMOTIONS:
        nrc_col = f"NRC_{emo}"
        if nrc_col not in merged.columns or emo not in merged.columns:
            rows.append(
                {"emotion": emo, "n": 0, "n11": 0, "n10": 0, "n01": 0, "n00": 0, "accuracy": np.nan}
            )
            continue
        nrc_vals = pd.to_numeric(merged[nrc_col], errors="coerce")
        gem_vals = pd.to_numeric(merged[emo], errors="coerce")
        mask = nrc_vals.notna() & gem_vals.notna()
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {"emotion": emo, "n": 0, "n11": 0, "n10": 0, "n01": 0, "n00": 0, "accuracy": np.nan}
            )
            continue
        nrc_bin = (nrc_vals[mask] > 0).astype(int)
        gem_bin = (gem_vals[mask] >= 0.5).astype(int)
        n11 = int(((nrc_bin == 1) & (gem_bin == 1)).sum())
        n10 = int(((nrc_bin == 1) & (gem_bin == 0)).sum())
        n01 = int(((nrc_bin == 0) & (gem_bin == 1)).sum())
        n00 = int(((nrc_bin == 0) & (gem_bin == 0)).sum())
        rows.append(
            {
                "emotion": emo,
                "n": n,
                "n11": n11,
                "n10": n10,
                "n01": n01,
                "n00": n00,
                "accuracy": float((n11 + n00) / n),
            }
        )
    return pd.DataFrame(rows)


def toplabel_agreement(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    out["gemini_primary_label"] = out.apply(_gemini_primary_label, axis=1)

    rows = []
    for col in MODEL_LABEL_COLS:
        if col not in out.columns:
            rows.append({"model": col, "n_compared": 0, "n_match": 0, "match_rate": np.nan})
            continue
        model_coarse = out[col].apply(_coarsen_label)
        mask = out["gemini_primary_label"].ne("multi") & out[col].notna()
        n_compared = int(mask.sum())
        n_match = int((model_coarse[mask] == out.loc[mask, "gemini_primary_label"]).sum())
        match_rate = float(n_match / n_compared) if n_compared else np.nan
        rows.append({"model": col, "n_compared": n_compared, "n_match": n_match, "match_rate": match_rate})
    return pd.DataFrame(rows)


def continuous_vad_agreement(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim in VAD_DIMS:
        u_col = f"VAD_{dim}"
        g_col = dim
        if u_col not in merged.columns or g_col not in merged.columns:
            rows.append(
                {
                    "dimension": dim,
                    "n": 0,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "spearman_r": np.nan,
                    "spearman_p": np.nan,
                    "mae": np.nan,
                }
            )
            continue
        x = pd.to_numeric(merged[u_col], errors="coerce")
        y = pd.to_numeric(merged[g_col], errors="coerce")
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "dimension": dim,
                    "n": 0,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "spearman_r": np.nan,
                    "spearman_p": np.nan,
                    "mae": np.nan,
                }
            )
            continue

        xv = x[mask].to_numpy(dtype=float)
        yv = y[mask].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(xv - yv)))

        if n >= 2 and np.std(xv) > 0 and np.std(yv) > 0:
            pear_r, pear_p = pearsonr(xv, yv)
            spear_r, spear_p = spearmanr(xv, yv)
        else:
            pear_r, pear_p = np.nan, np.nan
            spear_r, spear_p = np.nan, np.nan

        rows.append(
            {
                "dimension": dim,
                "n": n,
                "pearson_r": float(pear_r) if pd.notna(pear_r) else np.nan,
                "pearson_p": float(pear_p) if pd.notna(pear_p) else np.nan,
                "spearman_r": float(spear_r) if pd.notna(spear_r) else np.nan,
                "spearman_p": float(spear_p) if pd.notna(spear_p) else np.nan,
                "mae": mae,
            }
        )
    return pd.DataFrame(rows)


def run_analysis(
    *,
    unified_csv: Path,
    gemini_csv: Path,
    out_dir: Path,
    perfume_col: str = "Perfume",
    strict: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    unified = pd.read_csv(unified_csv)
    gemini = pd.read_csv(gemini_csv)

    unified_prepped = prepare_unified(unified, perfume_col=perfume_col)
    gemini_prepped = prepare_gemini(gemini)

    dup_unified = _duplicate_ids(unified_prepped, "perfume_id")
    dup_gemini = _duplicate_ids(gemini_prepped, "perfume_id")
    if strict and dup_unified:
        raise ValueError(f"Duplicate perfume_id in unified: {dup_unified[:10]}")
    if strict and dup_gemini:
        raise ValueError(f"Duplicate perfume_id in gemini: {dup_gemini[:10]}")
    if dup_unified:
        unified_prepped = unified_prepped.drop_duplicates(subset=["perfume_id"], keep="first")
    if dup_gemini:
        gemini_prepped = gemini_prepped.drop_duplicates(subset=["perfume_id"], keep="first")

    unified_only = unified_prepped.loc[
        ~unified_prepped["perfume_id"].isin(gemini_prepped["perfume_id"]),
        ["perfume_id", "perfume_id_hash_unified", "perfume_name_unified"],
    ].copy()
    gemini_only = gemini_prepped.loc[
        ~gemini_prepped["perfume_id"].isin(unified_prepped["perfume_id"]),
        ["perfume_id", "perfume_id_hash_gemini", "perfume_name_gemini"],
    ].copy()

    merged = unified_prepped.merge(gemini_prepped, on="perfume_id", how="inner", suffixes=("_unified", "_gemini"))

    hash_match_mask = (
        merged["perfume_id_hash_unified"].notna()
        & merged["perfume_id_hash_gemini"].notna()
        & merged["perfume_id_hash_unified"].astype(str).ne("")
        & merged["perfume_id_hash_gemini"].astype(str).ne("")
    )
    hash_mismatch_df = merged.loc[
        hash_match_mask
        & merged["perfume_id_hash_unified"].astype(str).ne(merged["perfume_id_hash_gemini"].astype(str)),
        ["perfume_id", "perfume_name_unified", "perfume_name_gemini", "perfume_id_hash_unified", "perfume_id_hash_gemini"],
    ].copy()
    if strict and not hash_mismatch_df.empty:
        raise ValueError(
            f"Hash mismatch detected for {len(hash_mismatch_df)} rows. "
            f"First IDs: {hash_mismatch_df['perfume_id'].head(5).tolist()}"
        )

    binary_agree = binary_nrc_agreement(merged)
    top_agree = toplabel_agreement(merged)
    vad_agree = continuous_vad_agreement(merged)

    merge_path = out_dir / "gemini_unified_merged.csv"
    join_report_path = out_dir / "gemini_unified_join_report.json"
    unmatched_unified_path = out_dir / "gemini_unified_unmatched_unified.csv"
    unmatched_gemini_path = out_dir / "gemini_unified_unmatched_gemini.csv"
    binary_path = out_dir / "agreement_binary_nrc_vs_gemini.csv"
    top_path = out_dir / "agreement_toplabel_models_vs_gemini.csv"
    vad_path = out_dir / "agreement_vad_continuous.csv"
    summary_path = out_dir / "gemini_unified_summary_report.json"

    merged.to_csv(merge_path, index=False)
    unified_only.to_csv(unmatched_unified_path, index=False)
    gemini_only.to_csv(unmatched_gemini_path, index=False)
    binary_agree.to_csv(binary_path, index=False)
    top_agree.to_csv(top_path, index=False)
    vad_agree.to_csv(vad_path, index=False)

    join_report = {
        "counts": {
            "unified_total": int(len(unified_prepped)),
            "gemini_total": int(len(gemini_prepped)),
            "matched": int(len(merged)),
            "unified_only": int(len(unified_only)),
            "gemini_only": int(len(gemini_only)),
        },
        "integrity": {
            "duplicate_ids_unified_count": int(len(dup_unified)),
            "duplicate_ids_gemini_count": int(len(dup_gemini)),
            "duplicate_ids_unified": dup_unified,
            "duplicate_ids_gemini": dup_gemini,
            "hash_mismatch_count": int(len(hash_mismatch_df)),
            "hash_mismatch_ids": hash_mismatch_df["perfume_id"].astype(str).tolist(),
        },
    }
    join_report_path.write_text(json.dumps(join_report, indent=2), encoding="utf-8")

    gemini_prev = {
        emo: float(pd.to_numeric(merged[emo], errors="coerce").mean()) if emo in merged.columns else np.nan
        for emo in EMOTIONS
    }
    gemini_vad_stats = {}
    for dim in VAD_DIMS:
        if dim in merged.columns:
            vals = pd.to_numeric(merged[dim], errors="coerce")
            gemini_vad_stats[dim] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
        else:
            gemini_vad_stats[dim] = {"mean": np.nan, "std": np.nan}

    summary_report = {
        "join_counts": join_report["counts"],
        "integrity": join_report["integrity"],
        "binary_agreement_mean_accuracy": float(binary_agree["accuracy"].mean(skipna=True)),
        "toplabel_match_rates": {
            str(r["model"]): (float(r["match_rate"]) if pd.notna(r["match_rate"]) else np.nan)
            for _, r in top_agree.iterrows()
        },
        "vad_continuous_agreement": vad_agree.to_dict(orient="records"),
        "gemini_prevalence": gemini_prev,
        "gemini_vad_stats": gemini_vad_stats,
    }
    summary_path.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")

    print(
        f"Matched {join_report['counts']['matched']} / unified={join_report['counts']['unified_total']} "
        f"/ gemini={join_report['counts']['gemini_total']}"
    )
    print(f"Wrote: {merge_path}")
    print(f"Wrote: {join_report_path}")
    print(f"Wrote: {binary_path}")
    print(f"Wrote: {top_path}")
    print(f"Wrote: {vad_path}")
    print(f"Wrote: {summary_path}")
    return summary_report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--unified-csv",
        type=Path,
        default=data_path("perfume_level", "Unified_Emotion_Profile_Perfume.csv"),
    )
    ap.add_argument(
        "--gemini-csv",
        type=Path,
        default=labels_path("gemini_labels_784_v1.csv"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=outputs_path("analysis_out"),
    )
    ap.add_argument("--perfume-col", type=str, default="Perfume")
    ap.add_argument("--strict", dest="strict", action="store_true", default=True)
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        unified_csv=args.unified_csv,
        gemini_csv=args.gemini_csv,
        out_dir=args.out_dir,
        perfume_col=args.perfume_col,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
