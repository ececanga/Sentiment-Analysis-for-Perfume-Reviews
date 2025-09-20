import os
import pandas as pd

csv_folder = "csv files"

print("🔍 Inspecting CSV headers in 'csv files/' folder...\n")

# Loop through all files in the folder
for filename in os.listdir(csv_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(csv_folder, filename)
        try:
            df = pd.read_csv(filepath, nrows=1)  # Only load the first row
            print(f"📄 {filename}:")
            print(f"   → Columns: {list(df.columns)}\n")
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}\n")
