"""apply_emotion_vad_lexicons.py
Purpose: Maps NRC emotion labels and VAD scores to perfumes using lexicons.
Input: csv files/YouTube_Perfume_Reviews_FullyFormatted.csv, NRC & VAD lexicons
Output:
  - csv files/NRC_Emotion_Labels_Perfume.csv
  - csv files/VAD_Scores_Perfume.csv
Execution Order: After formatting transcripts and before emotion analysis with transformers.
"""

import pandas as pd
import re
from collections import defaultdict

# --- Load your transcript dataset ---
df = pd.read_csv("csv files/YouTube_Perfume_Reviews_FullyFormatted.csv")

# --- Load NRC Emotion Lexicon ---
nrc = pd.read_csv(
    "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    sep="\t",
    header=None,
    names=["word", "emotion", "association"],
)
nrc = nrc[nrc["association"] == 1].drop("association", axis=1)
nrc_lexicon = nrc.groupby("word")["emotion"].apply(set).to_dict()

# --- Load VAD Lexicon ---
vad = pd.read_csv("NRC-VAD-Lexicon-v2.1.txt", sep="\t")
vad.columns = ["word", "valence", "arousal", "dominance"]
vad_lexicon = vad.set_index("word").T.to_dict()

# --- Define NRC emotion categories (10 classes) ---
emotion_categories = [
    "anger", "anticipation", "disgust", "fear", "joy", "sadness",
    "surprise", "trust", "positive", "negative"
]

# --- Initialize results ---
emotion_results = []
vad_results = []

# --- Process each perfume ---
for _, row in df.iterrows():
    perfume = row["Perfume"]
    tokens = re.findall(r"\b\w+\b", str(row["Final Transcript"]).lower())

    # Initialize counters
    emotion_counter = defaultdict(int)
    vad_values = {"valence": [], "arousal": [], "dominance": []}

    for word in tokens:
        # Count NRC emotions
        if word in nrc_lexicon:
            for emo in nrc_lexicon[word]:
                emotion_counter[emo] += 1

        # Collect VAD scores
        if word in vad_lexicon:
            vad_values["valence"].append(vad_lexicon[word]["valence"])
            vad_values["arousal"].append(vad_lexicon[word]["arousal"])
            vad_values["dominance"].append(vad_lexicon[word]["dominance"])

    # Final emotion + VAD results per perfume
    emotion_row = {"Perfume": perfume}
    emotion_row.update({e: emotion_counter[e] for e in emotion_categories})
    emotion_results.append(emotion_row)

    vad_row = {
        "Perfume": perfume,
        "valence": sum(vad_values["valence"]) / len(vad_values["valence"]) if vad_values["valence"] else 0,
        "arousal": sum(vad_values["arousal"]) / len(vad_values["arousal"]) if vad_values["arousal"] else 0,
        "dominance": sum(vad_values["dominance"]) / len(vad_values["dominance"]) if vad_values["dominance"] else 0,
    }
    vad_results.append(vad_row)

# --- Export Results ---
pd.DataFrame(emotion_results).to_csv("csv files/NRC_Emotion_Labels_Perfume.csv", index=False)
pd.DataFrame(vad_results).to_csv("csv files/VAD_Scores_Perfume.csv", index=False)
print("✅ Saved NRC_Emotion_Labels_Perfume.csv and VAD_Scores_Perfume.csv")
