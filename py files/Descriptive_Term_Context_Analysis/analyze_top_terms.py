'''analyze_top_terms.py
Purpose: Finds most frequent descriptive terms for each perfume
Input: perfume_context_keywords.csv
Output: Console output + optional perfume_intensifier_counts.csv'''

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOPWORDS

# --- Step 0: Load Context Matrix ---
df = pd.read_csv("perfume_context_keywords.csv", index_col=0)

# --- Step 1: Define Custom Stopwords, Scent/Emotion Terms, Intensifiers ---

# Custom Stopwords (non-descriptive fillers)
custom_stopwords = {
    "yeah", "guys", "little", "just", "know", "think", "don", "get", "going", "kind", 
    "sort", "thing", "okay", "want", "especially", "actually", "fragrance", 
    "perfume", "scent", "smell", "stuff", "comments", "ingredients", "read"
}

# Intensifiers (use for emotional weighting later)
intensifiers = {"really", "absolutely", "super", "very", "truly", "totally", "definitely", "incredibly"}

# Combine English stopwords and custom stopwords
all_stopwords = STOPWORDS.union(custom_stopwords)

# --- Step 2: Filter Descriptive Context Terms ---
filtered_cols = [
    col for col in df.columns 
    if col not in all_stopwords and col not in intensifiers and len(col) >= 3
]
df_filtered = df[filtered_cols]

# --- Step 3: Global Top Descriptive Terms ---
top_global = df_filtered.sum(axis=0).sort_values(ascending=False).head(30)
print("\n🧠 Top Global Scent-Emotion Context Terms (Filtered):")
print(top_global)

# --- Step 4: Top Descriptive Terms Per Perfume ---
print("\n🎯 Top 5 Descriptive Context Terms Per Perfume:")
for perfume in df_filtered.index:
    top_terms = df_filtered.loc[perfume].sort_values(ascending=False).head(5)
    print(f"- {perfume}: {', '.join(top_terms.index)}")

# --- Step 5: Save Intensifier Data for Later (Optional) ---
# Filter to only existing intensifiers
existing_intensifiers = [col for col in intensifiers if col in df.columns]

# Calculate intensifier counts per perfume
if existing_intensifiers:
    intensifier_counts = df[existing_intensifiers].sum(axis=1)
    intensifier_counts.name = "Intensifier Count"

    # Save as CSV for future emotional intensity analysis
    intensifier_counts.to_csv("perfume_intensifier_counts.csv")
    print("\n📊 Intensifier counts saved to perfume_intensifier_counts.csv")
else:
    print("\n⚠️ No intensifier columns found in data.")

