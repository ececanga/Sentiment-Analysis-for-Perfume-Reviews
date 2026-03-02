# 08) analyze_emotions_roberta_cardiffnlp.py
'''
Model: cardiffnlp/twitter-roberta-base-emotion
Purpose: Classifies into 4 emotions: joy, optimism, anger, sadness
Input: data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv
Output:
  - Per-review emotion scores and predicted label
  - Aggregated per-perfume emotion profile
'''

import pandas as pd

from src.utils.paths import data_path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

# Load model & tokenizer
model_name = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Load reviews
df = pd.read_csv(data_path("raw", "YouTube_Perfume_Reviews_FullyFormatted.csv"))

# Process each review
emotion_scores = []
predicted_emotions = []

print("🔍 Processing with cardiffnlp/twitter-roberta-base-emotion...")

for text in tqdm(df["Final Transcript"]):
    try:
        output = classifier(str(text)[:512])[0]  # Truncate text
        scores = {entry['label']: round(entry['score'], 4) for entry in output}
        top_emotion = max(scores, key=scores.get) if scores else None
    except Exception as e:
        print(f"⚠️ Error processing text: {e}")
        scores = {"joy": 0, "optimism": 0, "anger": 0, "sadness": 0}
        top_emotion = None  # safer fallback

    emotion_scores.append(scores)
    predicted_emotions.append(top_emotion)

# Create output DataFrame
scores_df = pd.DataFrame(emotion_scores)
scores_df["Perfume"] = df["Perfume"]
scores_df["Predicted Emotion"] = predicted_emotions

# Save per-review file
out_reviews = data_path("intermediate", "YouTube_Perfume_Reviews_With_CardiffRoberta.csv")
scores_df.to_csv(out_reviews, index=False)
print(f"✅ Saved per-review results: {out_reviews.name}")

# Aggregate per perfume
agg_df = scores_df.groupby("Perfume").mean(numeric_only=True).reset_index()
agg_df.to_csv(data_path("perfume_level", "Aggregated_CardiffRoberta_Perfume.csv"), index=False)
print("✅ Saved aggregated scores per perfume: Aggregated_CardiffRoberta_Perfume.csv")
