# Date Created: 2025-09-20 17:23:40+03:00
# Date Last Modified: 2026-05-15

''' 07) analyze_emotions_goemotions_multi.py
Model: Based on Google’s GoEmotions multilabel model
Purpose: Fine-grained emotion classification (e.g., approval, admiration, realization)
Input: data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv
Output: CSV with multi-label emotion probabilities per review and aggregated per perfume
'''

import pandas as pd

from src.utils.paths import data_path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch
from tqdm import tqdm

# Load model and tokenizer
model_name = "monologg/bert-base-cased-goemotions-original"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Detect device
device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
pipe = TextClassificationPipeline(
    model=model, tokenizer=tokenizer,
    return_all_scores=True, function_to_apply="sigmoid", device=device
)

# Load transcript data
df = pd.read_csv(data_path("raw", "YouTube_Perfume_Reviews_FullyFormatted.csv"))

# Get emotion label names from the model config
emotion_labels = model.config.id2label.values()

# Analyze each review
emotion_results = []
predicted_labels = []
for text in tqdm(df["Final Transcript"], desc="🔍 Processing GoEmotions"):
    outputs = pipe(str(text)[:512])[0]  # Truncate
    scores = {item["label"]: round(item["score"], 4) for item in outputs}
    top_label = max(scores, key=scores.get)
    emotion_results.append(scores)
    predicted_labels.append(top_label)

# Combine into DataFrame
emotion_df = pd.DataFrame(emotion_results)
emotion_df["Predicted Emotion"] = predicted_labels
emotion_df["Perfume"] = df["Perfume"]

# Save per-review outputs
out_reviews = data_path("intermediate", "YouTube_Perfume_Reviews_With_GoEmotions.csv")
emotion_df.to_csv(out_reviews, index=False)
print(f"✅ Saved to {out_reviews}")

# Aggregate per perfume (mean scores)
aggregated = emotion_df.groupby("Perfume").mean(numeric_only=True).reset_index()
out_perf = data_path("perfume_level", "Aggregated_GoEmotions_Perfume.csv")
aggregated.to_csv(out_perf, index=False)
print(f"✅ Saved unified emotion scores per perfume to {out_perf}")
