#!/usr/bin/env python3
# Date Created: 2026-03-02 11:07:58+03:00
# Date Last Modified: 2026-05-15

"""
create_silver_labels_fused.py

Goal
-----
Create *perfume-level* "silver" emotion labels using **score-based fusion** (not majority vote).

Why this exists
---------------
Majority vote over top labels collapses almost everything to "joy" in perfume reviews.
Instead, this script uses each model's *probability/score columns*, projects them into a shared
6-class space, **row-normalizes per model** (so GoEmotions doesn't dominate just because it has
more labels), then fuses models with equal (or user-defined) weights.

Input
-----
- data/perfume_level/Unified_Emotion_Profile_Perfume.csv
  Expected columns include:
    - Perfume
    - Hartmann_* scores (joy, neutral, surprise, sadness, anger, disgust, fear)
    - Savani_* scores (joy, sadness, love, anger, fear, surprise)
    - GoEmotions_* scores (admiration, amusement, ... , neutral)
    - Cardiff_* scores (joy, optimism, anger, sadness)

Output
------
- labels/silver/Unified_With_Silver_FusedLabels.csv
  Adds:
    - Fused_joy, Fused_sadness, Fused_anger, Fused_fear, Fused_surprise, Fused_neutral
    - Silver_Label_Fused
    - Silver_Confidence (max fused prob)
    - Silver_Margin (max - second max)

Run
---
python "src/labeling/create_silver_labels_fused.py"
python "src/labeling/create_silver_labels_fused.py" --input "data/perfume_level/Unified_Emotion_Profile_Perfume.csv" --output "labels/silver/Unified_With_Silver_FusedLabels.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.paths import data_path, labels_path

COARSE = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

# Fine label -> coarse bucket (shared space)
LABEL_MAP: Dict[str, str] = {
    # JOY / POSITIVE AFFECT
    "joy": "joy",
    "love": "joy",
    "optimism": "joy",
    "admiration": "joy",
    "approval": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "enthusiasm": "joy",
    "gratitude": "joy",
    "relief": "joy",
    "pride": "joy",
    "desire": "joy",
    "affection": "joy",
    "contentment": "joy",
    "happiness": "joy",
    "pleasure": "joy",
    "satisfaction": "joy",
    "trust": "joy",
    "positive": "joy",
    "caring": "joy",
    "curiosity": "joy",

    # SADNESS
    "sadness": "sadness",
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "loneliness": "sadness",
    "regret": "sadness",
    "sorrow": "sadness",
    "depression": "sadness",
    "melancholy": "sadness",

    # ANGER (including disgust-like)
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "anger",
    "contempt": "anger",
    "frustration": "anger",
    "irritation": "anger",
    "hatred": "anger",
    "resentment": "anger",
    "outrage": "anger",

    # FEAR / ANXIETY
    "fear": "fear",
    "nervousness": "fear",
    "anxiety": "fear",
    "worry": "fear",
    "apprehension": "fear",
    "panic": "fear",
    "dread": "fear",
    "insecurity": "fear",
    "embarrassment": "fear",
    "shame": "fear",

    # SURPRISE / REALIZATION / CONFUSION
    "surprise": "surprise",
    "astonishment": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "amazement": "surprise",
    "shock": "surprise",

    # NEUTRAL / OTHER
    "neutral": "neutral",
    "no emotion": "neutral",
    "other": "neutral",
    "others": "neutral",
    "unknown": "neutral",
}

# Model-specific fine labels to include (we read score columns and map suffixes via LABEL_MAP)
MODEL_PREFIXES = {
    "Hartmann": "Hartmann_",
    "Savani": "Savani_",
    "GoEmotions": "GoEmotions_",
    "Cardiff": "Cardiff_",
}


def norm_row(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    v[v < 0] = 0.0
    s = v.sum(axis=1, keepdims=True)
    s = np.where(s < eps, 1.0, s)
    return v / s


def collect_model_score_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def project_scores_to_coarse(df: pd.DataFrame, cols: List[str], prefix: str) -> pd.DataFrame:
    """
    For given model score columns, sum into coarse buckets, then row-normalize (per model).
    Returns DataFrame with columns: joy, sadness, anger, fear, surprise, neutral.
    """
    n = len(df)
    bucket = {k: np.zeros(n, dtype=float) for k in COARSE}

    for c in cols:
        # column name format: <Prefix><FineLabel>
        # e.g. GoEmotions_admiration => suffix is admiration
        suf = c.split(prefix, 1)[1].strip().lower()

        # normalize underscores/hyphens a bit
        suf_key = suf.replace("-", " ").strip()
        coarse = LABEL_MAP.get(suf_key, LABEL_MAP.get(suf_key.replace(" ", ""), None))

        if coarse in bucket:
            bucket[coarse] += pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    out = pd.DataFrame(bucket)
    out = out[COARSE]
    out = pd.DataFrame(norm_row(out.to_numpy()), columns=COARSE)
    return out


def fuse_models(
    df: pd.DataFrame,
    weights: Dict[str, float],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build per-model coarse distributions (row-normalized), then weighted average across models.
    Returns fused coarse probs and list of models actually used.
    """
    used_models = []
    fused = np.zeros((len(df), len(COARSE)), dtype=float)

    for model, pfx in MODEL_PREFIXES.items():
        w = float(weights.get(model, 0.0))
        if w <= 0:
            continue

        cols = collect_model_score_cols(df, pfx)
        if not cols:
            continue

        proj = project_scores_to_coarse(df, cols, pfx)  # row-normalized per model
        fused += w * proj.to_numpy(dtype=float)
        used_models.append(model)

    if not used_models:
        raise ValueError("No model score columns found. Check that your Unified CSV contains transformer score columns.")

    # renormalize after fusion
    fused = norm_row(fused)
    fused_df = pd.DataFrame(fused, columns=[f"Fused_{k}" for k in COARSE])
    return fused_df, used_models


def pick_label_and_conf(fused_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    M = fused_df.to_numpy(dtype=float)
    # idx of max and second max
    top_idx = np.argmax(M, axis=1)
    top_vals = M[np.arange(len(M)), top_idx]

    # second max
    part = np.partition(M, -2, axis=1)
    second_vals = part[:, -2]

    labels = pd.Series([COARSE[i] for i in top_idx], name="Silver_Label_Fused")
    conf = pd.Series(top_vals, name="Silver_Confidence")
    margin = pd.Series(top_vals - second_vals, name="Silver_Margin")
    return labels, conf, margin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=data_path("perfume_level", "Unified_Emotion_Profile_Perfume.csv"),
        help="Input unified perfume CSV",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=labels_path("silver", "Unified_With_Silver_FusedLabels.csv"),
        help="Output CSV path",
    )
    ap.add_argument(
        "--w_hartmann", type=float, default=1.0,
        help="Weight for Hartmann_* scores in fusion",
    )
    ap.add_argument(
        "--w_savani", type=float, default=1.0,
        help="Weight for Savani_* scores in fusion",
    )
    ap.add_argument(
        "--w_goemotions", type=float, default=1.0,
        help="Weight for GoEmotions_* scores in fusion",
    )
    ap.add_argument(
        "--w_cardiff", type=float, default=1.0,
        help="Weight for Cardiff_* scores in fusion",
    )
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    df = pd.read_csv(in_path)

    if "Perfume" not in df.columns:
        raise ValueError("Expected column 'Perfume' in input CSV.")

    # Ensure uniqueness at perfume level (your pipeline should already be unique, but safe here)
    df = df.drop_duplicates(subset=["Perfume"]).reset_index(drop=True)

    weights = {
        "Hartmann": float(args.w_hartmann),
        "Savani": float(args.w_savani),
        "GoEmotions": float(args.w_goemotions),
        "Cardiff": float(args.w_cardiff),
    }

    fused_df, used_models = fuse_models(df, weights=weights)
    labels, conf, margin = pick_label_and_conf(fused_df)

    out = df.copy()
    out = pd.concat([out, fused_df, labels, conf, margin], axis=1)

    out.to_csv(out_path, index=False)

    print("✅ Fused silver labels created.")
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"Models used: {', '.join(used_models)}")
    print("\nLabel distribution:")
    print(out["Silver_Label_Fused"].value_counts())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
