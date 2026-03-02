import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.paths import PROJECT_ROOT, labels_path

EMO = ["joy", "sadness", "anger", "fear", "trust", "surprise", "anticipation", "disgust"]
VAD = ["valence", "arousal", "dominance"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=labels_path("llm", "strict", "labels_strict_v1.json"),
        help="Path to labels JSON (default: labels/llm/strict/labels_strict_v1.json)",
    )
    args = ap.parse_args()

    in_json = args.input
    if not in_json.is_absolute():
        in_json = PROJECT_ROOT / in_json

    if not in_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_json}")

    data = json.loads(in_json.read_text(encoding="utf-8"))
    df = pd.DataFrame.from_dict(data, orient="index").reset_index().rename(columns={"index": "Perfume"})

    print("\n=== Emotion positive counts ===")
    print(df[EMO].sum().sort_values(ascending=False))

    print("\n=== Emotion positive rates ===")
    print((df[EMO].mean()).sort_values(ascending=False).round(3))

    print("\n=== VAD summary ===")
    print(df[VAD].describe().round(3))

    print("\n=== Correlations (VAD) ===")
    print(df[VAD].corr().round(3))


if __name__ == "__main__":
    main()
