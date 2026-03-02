''' 05) analyze_emotions_distilroberta_jhartmann.py
Model: j-hartmann/emotion-english-distilroberta-base
Purpose: Categorical emotion detection (joy, sadness, anger, etc.)
Input: data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv
Output: data/intermediate/YouTube_Perfume_Reviews_With_Transformer.csv'''


# emotion_transformer_inference.py

import pandas as pd

from src.utils.paths import data_path
from transformers import pipeline
from tqdm import tqdm

# Load formatted transcripts
df = pd.read_csv(data_path("raw", "YouTube_Perfume_Reviews_FullyFormatted.csv"))

# Load emotion classification pipeline
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    top_k=None
)

# For progress tracking
tqdm.pandas()

# Inference function
def classify_emotions(text):
    try:
        results = emotion_classifier(text[:512])  # Truncate to 512 tokens
        sorted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)
        return {entry["label"]: round(entry["score"], 4) for entry in sorted_results}
    except Exception as e:
        return {}

# Apply to Final Transcript column
df["Transformer_Emotion_Distribution"] = df["Final Transcript"].progress_apply(classify_emotions)

# Optionally: extract top-1 label
df["Transformer_Top_Emotion"] = df["Transformer_Emotion_Distribution"].apply(
    lambda d: max(d, key=d.get) if isinstance(d, dict) and d else None
)

# Save
out_path = data_path("intermediate", "YouTube_Perfume_Reviews_With_Transformer.csv")
df.to_csv(out_path, index=False)
print(f"✅ Saved to {out_path.name}")
