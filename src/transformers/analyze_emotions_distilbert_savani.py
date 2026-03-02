# 06) analyze_emotions_distilbert_savani.py
# Model: bhadresh-savani/distilbert-base-uncased-emotion
# Purpose: Alternate transformer-based classification (adds Predicted Emotion)
# Input: data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv
# Output: data/intermediate/YouTube_Perfume_Reviews_With_Savani.csv

import pandas as pd

from src.utils.paths import data_path
from transformers import pipeline
from tqdm import tqdm
import torch

# Device setup
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

# Load model
classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    top_k=None,
    device=device
)

# Load transcripts
df = pd.read_csv(data_path("raw", "YouTube_Perfume_Reviews_FullyFormatted.csv"))

# Prepare output list
output_rows = []

# Run inference on each transcript
for idx, row in tqdm(df.iterrows(), total=len(df)):
    perfume = row["Perfume"]
    text = str(row["Final Transcript"])

    try:
        # Truncate text (approx chars; good enough for minimal change)
        result = classifier(text[:512])[0]
        # Normalize labels to lowercase and keep scores
        emotion_scores = {entry["label"].lower(): float(entry["score"]) for entry in result}
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        continue

    emotion_scores["Perfume"] = perfume
    output_rows.append(emotion_scores)

# Create DataFrame from results
results_df = pd.DataFrame(output_rows)

# Ensure consistent column order if available
ordered_cols = ["Perfume", "joy", "sadness", "love", "anger", "fear", "surprise"]
existing_cols = [c for c in ordered_cols if c in results_df.columns]
results_df = results_df[existing_cols + [c for c in results_df.columns if c not in existing_cols]]

# Add Predicted Emotion via idxmax across emotion score columns
score_cols = [c for c in results_df.columns if c not in {"Perfume"}]
if score_cols:
    results_df["Predicted Emotion"] = results_df[score_cols].idxmax(axis=1)
else:
    results_df["Predicted Emotion"] = None

# Save to CSV (path matches your merge script)
out_path = data_path("intermediate", "YouTube_Perfume_Reviews_With_Savani.csv")
results_df.to_csv(out_path, index=False)
print(f"✅ Saved transformer emotion outputs to {out_path}")
