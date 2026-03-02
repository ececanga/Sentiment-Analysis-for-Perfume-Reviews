# 12) merge_emotion_profiles.py

import pandas as pd

# ---- Load base files from your folder ----
nrc_df = pd.read_csv("csv files/NRC_Emotion_Labels_Perfume.csv")
vad_df = pd.read_csv("csv files/VAD_Scores_Perfume.csv")

# ---- Aggregate to one row per Perfume BEFORE merging ----
# NRC: sum counts per perfume
nrc_agg = nrc_df.groupby("Perfume", as_index=False).sum(numeric_only=True)

# VAD: mean per perfume
vad_agg = vad_df.groupby("Perfume", as_index=False).mean(numeric_only=True)

# Merge aggregated NRC + VAD (prevents many-to-many explosion)
merged_df = pd.merge(nrc_agg, vad_agg, on="Perfume", how="inner")

# ---- Transformer configs (paths unchanged) ----
transformers = {
    "JHartmann": {
        "path": "csv files/YouTube_Perfume_Reviews_With_Transformer.csv",
        "label_col": "Transformer_Top_Emotion"
    },
    "GoEmotions": {
        "path": "csv files/YouTube_Perfume_Reviews_With_GoEmotions.csv",
        "label_col": "Predicted Emotion"
    },
    "Cardiff": {
        "path": "csv files/YouTube_Perfume_Reviews_With_CardiffRoberta.csv",
        "label_col": "Predicted Emotion"
    },
    "Savani": {
        "path": "csv files/YouTube_Perfume_Reviews_With_Savani.csv",
        "label_col": "Predicted Emotion"  # will be skipped if not present
    }
}

for label, config in transformers.items():
    try:
        df = pd.read_csv(config["path"])
    except FileNotFoundError:
        print(f"❌ Missing file for {label}: {config['path']} — skipping…")
        continue

    print(f"\n📄 [{label}] Columns in {config['path']}:")
    print(df.columns.tolist())  # Debug output

    if config["label_col"] not in df.columns:
        print(f"❌ Column '{config['label_col']}' not found in {label} — skipping…")
        continue

    # Collapse to one categorical label per perfume via mode (most frequent)
    label_counts = df.groupby("Perfume")[config["label_col"]].agg(
        lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
    )
    label_counts.name = f"{label}_Label"

    merged_df = pd.merge(merged_df, label_counts, on="Perfume", how="left")

# Save final unified profile
merged_df.to_csv("csv files/Unified_Emotion_Profile_Perfume.csv", index=False)
print("✅ Saved merged profile with NRC, VAD, and Transformer outputs to csv files/Unified_Emotion_Profile_Perfume.csv")
