from pathlib import Path

import pandas as pd

from src.utils.paths import data_path

csv_folder = data_path()

print(f"🔍 Inspecting CSV headers in {csv_folder}/\n")

# Loop through all CSV files under data/
for filepath in sorted(csv_folder.rglob("*.csv")):
    try:
        df = pd.read_csv(filepath, nrows=1)  # Only load the first row
        rel = filepath.relative_to(csv_folder)
        print(f"📄 {rel}:")
        print(f"   → Columns: {list(df.columns)}\n")
    except Exception as e:
        print(f"⚠️ Error reading {filepath}: {e}\n")
