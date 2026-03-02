# 09) extract_context_terms.py
# Purpose: Uses emotion-labeled words as anchors to extract context windows
# Input: data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv
# Output: perfume_context_keywords.csv

import pandas as pd

from src.utils.paths import data_path
import re
from collections import Counter

# Load cleaned transcripts
df = pd.read_csv(data_path("raw", "YouTube_Perfume_Reviews_FullyFormatted.csv"))

# Emotion terms (unchanged list)
emotion_terms = {
    "love", "adore", "worship", "obsessed", "devoted", "intoxicated", "infatuated", "captivated",
    "mesmerized", "seduced", "passion", "burning", "soulful", "spellbound",
    "beautiful", "sublime", "exquisite", "magnificent", "elegant", "regal", "majestic",
    "graceful", "luxurious", "divine", "heavenly", "angelic", "ethereal", "breathtaking",
    "joyful", "cheerful", "uplifting", "playful", "sparkling", "radiant", "delightful",
    "vivacious", "sunny", "luminous", "bright", "fresh", "youthful", "effervescent",
    "elated", "happy",
    "calming", "soothing", "relaxing", "serene", "tranquil", "peaceful", "cozy", "gentle",
    "soft", "meditative", "zen", "comforting", "harmonious", "whispering",
    "mysterious", "enigmatic", "dark", "brooding", "moody", "haunting", "melancholic",
    "deep", "sultry", "intense", "hypnotic", "noir", "gothic", "shadowy", "voluptuous",
    "nostalgic", "sentimental", "timeless", "vintage", "classic", "old-school", "familiar",
    "heartwarming", "homey", "reminiscent", "childhood", "memory-laden", "retro",
    "exotic", "rare", "unusual", "eccentric", "avant-garde", "whimsical", "curious", "quirky",
    "adventurous", "daring", "bold", "unexpected", "surprising", "innovative", "visionary",
    "strong", "assertive", "confident", "empowering", "fearless", "intrepid",
    "commanding", "dominant", "decisive", "striking", "ferocious", "heroic", "vigorous",
    "relentless",
    "hate", "disappointed", "repulsive", "cloying", "overwhelming", "harsh", "chemical",
    "synthetic", "awkward", "too sweet", "too sharp", "boring", "flat", "uninspired", "weird"
}

# --- Minimal fixes ---

# 1) Handle multi-word / hyphenated anchors
phrase_terms = {t.replace("-", " ") for t in emotion_terms if " " in t or "-" in t}
single_terms = {t for t in emotion_terms if t not in phrase_terms and " " not in t and "-" not in t}

# 2) Tiny stopword filter and min-length to reduce junk context
STOPWORDS = {
    "the","a","an","and","or","but","if","so","to","of","in","on","for","as","at",
    "with","is","it","this","that","these","those","be","are","was","were","by",
    "from","you","i","we","they","he","she","me","my","your","our","their"
}
MIN_LEN = 3
WINDOW = 4
TOP_CONTEXT_PER_PERFUME = 200  # cap width per perfume

# Tokenize to words
df["Tokenized"] = df["Final Transcript"].apply(lambda x: re.findall(r"\b\w+\b", str(x).lower()))

context_matrix = {}

for _, row in df.iterrows():
    tokens = row["Tokenized"]
    perfume = row["Perfume"]

    # Build bigrams for phrase matching
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]

    if perfume not in context_matrix:
        context_matrix[perfume] = Counter()

    i = 0
    while i < len(tokens):
        word = tokens[i]

        # Check phrase anchor first (bigram), then single-word anchor
        is_anchor = False
        if i < len(tokens)-1 and bigrams[i] in phrase_terms:
            is_anchor = True
            span_after = 2
        elif word in single_terms:
            is_anchor = True
            span_after = 1

        if is_anchor:
            left = tokens[max(0, i-WINDOW): i]
            right = tokens[i+span_after: i+span_after+WINDOW]
            for cw in left + right:
                if len(cw) >= MIN_LEN and cw not in STOPWORDS:
                    context_matrix[perfume][cw] += 1
            i += span_after  # skip consumed tokens for bigram
        else:
            i += 1

# Cap width per perfume to keep CSV manageable
for p in list(context_matrix.keys()):
    context_matrix[p] = dict(context_matrix[p].most_common(TOP_CONTEXT_PER_PERFUME))

# Convert to DataFrame
context_df = pd.DataFrame.from_dict(context_matrix, orient="index").fillna(0).astype(int)

# Save
context_df.to_csv(data_path("perfume_level", "perfume_context_keywords.csv"))
print("✅ Saved to perfume_context_keywords.csv")
