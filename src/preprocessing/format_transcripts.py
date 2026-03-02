""" 03) format_transcripts.py
Purpose: Cleans and formats transcripts (e.g., removes extra whitespace, newlines, formatting errors).
Input: data/raw/YouTube_Perfume_Reviews_Partial.csv
Output: data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv
Execution Order: Run after scraping """

import pandas as pd
import sys
import csv
import re

from src.utils.paths import data_path
# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

# Now safely load the large CSV file
df = pd.read_csv(data_path("raw", "YouTube_Perfume_Reviews_Partial.csv"), engine="python")


# Clean each transcript entry: remove extra spaces and flatten lines
def fix_transcript_format(text):
    text = str(text)
    text = re.sub(r'\s{3,}', ' ', text)  # Replace 3+ spaces with single space
    text = text.replace('\n', ' ')       # Remove any newline characters inside fields
    text = re.sub(r'\s{2,}', ' ', text)  # Normalize again: 2+ spaces to 1
    return text.strip()

# Apply to Final Transcript column (and optionally Transcript column)
df['Final Transcript'] = df['Final Transcript'].apply(fix_transcript_format)
df['Transcript'] = df['Transcript'].apply(fix_transcript_format)

# Save the fully cleaned and correctly formatted CSV
out_path = data_path("raw", "YouTube_Perfume_Reviews_FullyFormatted.csv")
df.to_csv(out_path, index=False)

print(f"✅ Saved cleaned and CSV-formatted file as {out_path.name}")
