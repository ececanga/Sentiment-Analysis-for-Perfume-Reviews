# Date Created: 2026-06-14
# Date Last Modified: 2026-06-14

"""Validate and merge browser LLM review-level labeling output."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.labeling.strict_schema import EMOTIONS, VAD_DIMS
from src.utils.paths import PROJECT_ROOT, data_path, labels_path

OUTPUT_COLUMNS = [
    "synthetic_review_id",
    "source_type",
    "perfume_id",
    "perfume_name",
    "review_index",
    *EMOTIONS,
    *VAD_DIMS,
]

PIPELINE_COLUMNS = [
    "synthetic_review_id",
    "perfume_id",
    "perfume_name",
    *EMOTIONS,
    *VAD_DIMS,
]


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_pasted_outputs(path: Path) -> list[tuple[int, pd.DataFrame]]:
    text = path.read_text(encoding="utf-8").strip()
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty pasted output: {path}")

    batch_starts: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        batch_match = re.match(r"^Batch\s+(\d+)\s+output:\s*$", line.strip(), flags=re.IGNORECASE)
        if batch_match:
            batch_starts.append((index, int(batch_match.group(1))))

    if not batch_starts or batch_starts[0][0] != 0:
        raise ValueError("First non-empty line must be like: Batch 1 output:")

    parsed: list[tuple[int, pd.DataFrame]] = []
    for start_pos, (line_index, batch_number) in enumerate(batch_starts):
        next_line_index = batch_starts[start_pos + 1][0] if start_pos + 1 < len(batch_starts) else len(lines)
        csv_text = "\n".join(lines[line_index + 1 : next_line_index])
        reader = csv.DictReader(io.StringIO(csv_text))
        if reader.fieldnames != OUTPUT_COLUMNS:
            raise ValueError(f"Unexpected CSV header for batch {batch_number}: {reader.fieldnames}. Expected: {OUTPUT_COLUMNS}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"No CSV rows found after the batch {batch_number} tracking line.")
        parsed.append((batch_number, pd.DataFrame(rows, columns=OUTPUT_COLUMNS)))

    batch_numbers = [batch_number for batch_number, _ in parsed]
    if len(batch_numbers) != len(set(batch_numbers)):
        raise ValueError(f"Duplicate batch sections found: {batch_numbers}")

    return parsed


def expected_for_batch(manifest: pd.DataFrame, batch_number: int) -> pd.DataFrame:
    prefix = f"v2_review_batch_{batch_number:02d}_"
    out = manifest[manifest["batch_file"].astype(str).str.startswith(prefix)].copy()
    if out.empty:
        raise ValueError(f"No manifest rows found for batch {batch_number}.")
    return out


def validate_rows(df: pd.DataFrame, expected: pd.DataFrame, batch_number: int) -> None:
    if len(df) != len(expected):
        raise ValueError(f"Batch {batch_number} has {len(df)} rows; expected {len(expected)}.")
    if not df["synthetic_review_id"].is_unique:
        raise ValueError(f"Batch {batch_number} contains duplicate synthetic_review_id values.")

    expected_ids = expected["synthetic_review_id"].astype(str).tolist()
    actual_ids = df["synthetic_review_id"].astype(str).tolist()
    missing = [value for value in expected_ids if value not in set(actual_ids)]
    extra = [value for value in actual_ids if value not in set(expected_ids)]
    if missing or extra:
        raise ValueError(f"Batch {batch_number} ID mismatch. Missing={missing}; extra={extra}")
    if actual_ids != expected_ids:
        raise ValueError(f"Batch {batch_number} row order differs from manifest order.")

    expected_lookup = expected.set_index("synthetic_review_id")
    for _, row in df.iterrows():
        source_id = row["synthetic_review_id"]
        exp = expected_lookup.loc[source_id]
        for column in ["source_type", "perfume_id", "perfume_name"]:
            if str(row[column]) != str(exp[column]):
                raise ValueError(f"{source_id} has wrong {column}: {row[column]} != {exp[column]}")
        if int(row["review_index"]) != int(exp["review_index"]):
            raise ValueError(f"{source_id} has wrong review_index: {row['review_index']} != {exp['review_index']}")

    for column in EMOTIONS:
        values = set(df[column].astype(str))
        if not values <= {"0", "1"}:
            raise ValueError(f"{column} must be binary 0/1, found {sorted(values)}")
        df[column] = df[column].astype(int)

    for column in VAD_DIMS:
        numeric = pd.to_numeric(df[column], errors="raise")
        bad = numeric[~numeric.between(1, 9) | (numeric % 1 != 0)]
        if not bad.empty:
            raise ValueError(f"{column} must be integer 1-9, found {bad.tolist()}")
        df[column] = numeric.astype(int)

    df["review_index"] = df["review_index"].astype(int)


def write_outputs(df: pd.DataFrame, pasted_path: Path, manifest: pd.DataFrame, batch_number: int, out_dir: Path, output_tag: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_batches"
    batch_dir = out_dir / "batch_csv"
    raw_dir.mkdir(exist_ok=True)
    batch_dir.mkdir(exist_ok=True)

    raw_path = raw_dir / f"{output_tag}_batch_{batch_number:03d}.txt"
    batch_csv_path = batch_dir / f"{output_tag}_batch_{batch_number:03d}.csv"
    combined_csv_path = out_dir / f"{output_tag}.csv"
    pipeline_csv_path = out_dir / f"{output_tag}_pipeline.csv"
    report_path = out_dir / f"{output_tag}_report.json"

    raw_path.write_text(pasted_path.read_text(encoding="utf-8").strip() + "\n", encoding="utf-8")
    df.to_csv(batch_csv_path, index=False)

    batch_frames = []
    for path in sorted(batch_dir.glob(f"{output_tag}_batch_*.csv")):
        batch_frames.append(pd.read_csv(path))
    combined = pd.concat(batch_frames, ignore_index=True) if batch_frames else df.copy()
    combined["_manifest_order"] = combined["synthetic_review_id"].map(
        {sid: idx for idx, sid in enumerate(manifest["synthetic_review_id"].astype(str).tolist())}
    )
    combined = combined.sort_values("_manifest_order").drop(columns=["_manifest_order"]).reset_index(drop=True)

    if combined["synthetic_review_id"].duplicated().any():
        duplicated = combined.loc[combined["synthetic_review_id"].duplicated(), "synthetic_review_id"].tolist()
        raise ValueError(f"Combined output contains duplicate IDs: {duplicated}")

    combined.to_csv(combined_csv_path, index=False)
    combined[PIPELINE_COLUMNS].to_csv(pipeline_csv_path, index=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_tag": output_tag,
        "last_ingested_batch": batch_number,
        "completed_rows": int(len(combined)),
        "completed_batches": int(len(batch_frames)),
        "target_rows": int(len(manifest)),
        "remaining_rows": int(len(manifest) - len(combined)),
        "emotion_positive_counts": {column: int(combined[column].sum()) for column in EMOTIONS},
        "vad_summary": {
            column: {
                "mean": float(combined[column].mean()),
                "min": int(combined[column].min()),
                "max": int(combined[column].max()),
            }
            for column in VAD_DIMS
        },
        "outputs": {
            "raw_batch": stringify_path(raw_path),
            "batch_csv": stringify_path(batch_csv_path),
            "combined_csv": stringify_path(combined_csv_path),
            "pipeline_csv": stringify_path(pipeline_csv_path),
            "report": stringify_path(report_path),
        },
        "note": "Browser Gemini pseudo-labels for emotion-rich synthetic v2, 3 reviews per perfume.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest browser LLM review-level label output.")
    parser.add_argument("--pasted-text", type=Path, required=True)
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=data_path("annotation_packets", "emotion_rich_v2_3reviews_gemini_browser_review_level", "manifest.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=labels_path("gemini_browser_review_level"))
    parser.add_argument("--output-tag", default="gemini_browser_labels_emotion_rich_v2_3reviews_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv)
    parsed_outputs = parse_pasted_outputs(args.pasted_text)
    report = None
    for batch_number, df in parsed_outputs:
        expected = expected_for_batch(manifest, batch_number)
        validate_rows(df, expected, batch_number)
        report = write_outputs(df, args.pasted_text, manifest, batch_number, args.out_dir, args.output_tag)
        print(f"Ingested batch {batch_number}: {len(df)} rows")
    assert report is not None
    print(f"Combined rows: {report['completed_rows']} / {report['target_rows']}")
    print(f"Combined CSV: {report['outputs']['combined_csv']}")
    print(f"Pipeline CSV: {report['outputs']['pipeline_csv']}")


if __name__ == "__main__":
    main()
