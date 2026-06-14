# Date Created: 2025-09-20 17:23:40+03:00
# Date Last Modified: 2026-05-15

"""Normalize transcript formatting for raw YouTube perfume review CSVs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

from src.utils.paths import data_path

csv.field_size_limit(sys.maxsize)


def fix_transcript_format(text: object) -> str:
    text = str(text)
    text = re.sub(r"\s{3,}", " ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def format_transcripts(input_csv: Path, output_csv: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input transcript CSV not found: {input_csv}")
    df = pd.read_csv(input_csv, engine="python")
    required = {"Transcript", "Final Transcript"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required transcript columns in {input_csv}: {missing}")

    df["Final Transcript"] = df["Final Transcript"].apply(fix_transcript_format)
    df["Transcript"] = df["Transcript"].apply(fix_transcript_format)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned and CSV-formatted file as {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize whitespace in YouTube perfume transcript CSVs.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=data_path("raw", "YouTube_Perfume_Reviews_Partial.csv"),
        help="Input raw/partial transcript CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=data_path("raw", "YouTube_Perfume_Reviews_FullyFormatted.csv"),
        help="Output formatted transcript CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    format_transcripts(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
