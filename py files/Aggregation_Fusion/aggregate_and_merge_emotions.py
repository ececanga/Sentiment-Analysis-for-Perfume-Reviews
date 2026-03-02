# 13) aggregate_and_merge_emotions.py
# Purpose: Aggregate NRC, VAD, and ALL transformer score columns into one per-perfume profile.
# Inputs (under "csv files/"):
#   - NRC_Emotion_Labels_Perfume.csv
#   - VAD_Scores_Perfume.csv
#   - YouTube_Perfume_Reviews_With_Transformer.csv        (Hartmann; has dict column)
#   - YouTube_Perfume_Reviews_With_Savani.csv             (Savani; flat columns)
#   - YouTube_Perfume_Reviews_With_GoEmotions.csv         (GoEmotions; flat columns)
#   - YouTube_Perfume_Reviews_With_CardiffRoberta.csv     (Cardiff; flat columns)
# Output:
#   - csv files/Unified_Emotion_Profile_Perfume.csv

import pandas as pd
import ast

# ----------------- Helpers -----------------
META_COLS = {"Perfume", "Predicted Emotion", "Transformer_Top_Emotion", "Final Transcript", "Transcript"}

def agg_numeric_means(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Keep numeric score columns (excluding META_COLS), average per Perfume, and prefix col names.
    Returns a DataFrame with columns: Perfume, <prefix>_<score1>, ...
    """
    if "Perfume" not in df.columns:
        raise ValueError(f"{prefix}: 'Perfume' column is missing.")

    # Auto-detect numeric score columns
    numeric_cols = [c for c in df.columns
                    if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols:
        # Nothing numeric to aggregate; return just Perfume to allow safe merge (will be dropped later)
        return df[["Perfume"]].drop_duplicates()

    agg = (df[["Perfume"] + numeric_cols]
           .groupby("Perfume", as_index=False)
           .mean(numeric_only=True))

    # Prefix columns to avoid collisions
    rename_map = {c: f"{prefix}_{c}" for c in numeric_cols}
    agg = agg.rename(columns=rename_map)
    return agg

def hartmann_expand_and_agg(df: pd.DataFrame, dict_col: str = "Transformer_Emotion_Distribution",
                            prefix: str = "Hartmann") -> pd.DataFrame:
    if "Perfume" not in df.columns:
        raise ValueError("Hartmann: 'Perfume' column is missing.")

    if dict_col in df.columns:
        def parse_dict(x):
            if isinstance(x, dict): return x
            if isinstance(x, str):
                try: return ast.literal_eval(x)
                except Exception: return {}
            return {}

        expanded = pd.json_normalize(df[dict_col].apply(parse_dict))
        if expanded is not None and not expanded.empty:
            expanded.columns = [f"{prefix}_{c}" for c in expanded.columns]
            tmp = pd.concat([df[["Perfume"]], expanded], axis=1)
            return (tmp.groupby("Perfume", as_index=False)
                        .mean(numeric_only=True))

    # Fallback: average any numeric cols if the dict column is missing
    return agg_numeric_means(df, prefix=prefix)

# ----------------- Load & aggregate NRC/VAD -----------------
nrc_df = pd.read_csv("csv files/NRC_Emotion_Labels_Perfume.csv")
vad_df = pd.read_csv("csv files/VAD_Scores_Perfume.csv")

# NRC: counts -> sum per perfume
nrc_numeric = nrc_df.select_dtypes(include="number").columns.tolist()
nrc_agg = (nrc_df[["Perfume"] + nrc_numeric]
           .groupby("Perfume", as_index=False)
           .sum(numeric_only=True))
# Prefix NRC columns for clarity
nrc_agg = nrc_agg.rename(columns={c: f"NRC_{c}" for c in nrc_numeric})

# VAD: continuous -> mean per perfume
vad_numeric = vad_df.select_dtypes(include="number").columns.tolist()
vad_agg = (vad_df[["Perfume"] + vad_numeric]
           .groupby("Perfume", as_index=False)
           .mean(numeric_only=True))
vad_agg = vad_agg.rename(columns={c: f"VAD_{c}" for c in vad_numeric})

# ----------------- Load & aggregate Transformers -----------------
# Hartmann (dict column)
hartmann_raw = pd.read_csv("csv files/YouTube_Perfume_Reviews_With_Transformer.csv")
hartmann_agg = hartmann_expand_and_agg(hartmann_raw, dict_col="Transformer_Emotion_Distribution", prefix="Hartmann")

# Savani (flat columns + Predicted Emotion)
savani_raw = pd.read_csv("csv files/YouTube_Perfume_Reviews_With_Savani.csv")
savani_agg = agg_numeric_means(savani_raw, prefix="Savani")

# GoEmotions (flat columns + Predicted Emotion)
goemo_raw = pd.read_csv("csv files/YouTube_Perfume_Reviews_With_GoEmotions.csv")
goemo_agg = agg_numeric_means(goemo_raw, prefix="GoEmotions")

# Cardiff (flat columns + Predicted Emotion)
cardiff_raw = pd.read_csv("csv files/YouTube_Perfume_Reviews_With_CardiffRoberta.csv")
cardiff_agg = agg_numeric_means(cardiff_raw, prefix="Cardiff")

# ----------------- Merge all -----------------
merged = pd.merge(nrc_agg, vad_agg, on="Perfume", how="inner")

# Left-join transformer aggregates so we don't drop perfumes lacking some models
for tf_df in [hartmann_agg, savani_agg, goemo_agg, cardiff_agg]:
    # If the transformer DataFrame has only 'Perfume' (no numeric cols), skip to avoid adding nothing
    if list(tf_df.columns) == ["Perfume"]:
        continue
    merged = pd.merge(merged, tf_df, on="Perfume", how="left")

# ---- Add categorical top labels (mode per perfume) ----
label_sources = {
    "JHartmann_Label": ("csv files/YouTube_Perfume_Reviews_With_Transformer.csv", "Transformer_Top_Emotion"),
    "Savani_Label":    ("csv files/YouTube_Perfume_Reviews_With_Savani.csv", "Predicted Emotion"),
    "GoEmotions_Label":("csv files/YouTube_Perfume_Reviews_With_GoEmotions.csv", "Predicted Emotion"),
    "Cardiff_Label":   ("csv files/YouTube_Perfume_Reviews_With_CardiffRoberta.csv", "Predicted Emotion"),
}

for out_col, (path, label_col) in label_sources.items():
    try:
        _df = pd.read_csv(path)
        if {"Perfume", label_col}.issubset(_df.columns):
            mode_per_perf = _df.groupby("Perfume")[label_col].agg(
                lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
            )
            mode_per_perf.name = out_col
            merged = pd.merge(merged, mode_per_perf, on="Perfume", how="left")
        else:
            print(f"⚠️ {path} missing Perfume/{label_col}; skipped label join for {out_col}")
    except FileNotFoundError:
        print(f"⚠️ Missing per-review file: {path} — skipped {out_col}")

# ----------------- Save -----------------
out_path = "csv files/Unified_Emotion_Profile_Perfume.csv"
merged.to_csv(out_path, index=False)
print(f"✅ Saved aggregated NRC, VAD, and all transformer scores to {out_path}")
