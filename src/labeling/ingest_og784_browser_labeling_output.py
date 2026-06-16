# Date Created: 2026-06-15
# Date Last Modified: 2026-06-15

"""Validate and merge browser Gemini outputs for OG 784 perfume labels."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.labeling.prepare_og784_browser_labeling_packets import OUTPUT_COLUMNS
from src.labeling.strict_schema import EMOTIONS, VAD_DIMS
from src.utils.paths import PROJECT_ROOT, data_path, labels_path

PIPELINE_COLUMNS = [
    "perfume_id",
    "perfume_id_hash",
    "perfume_name",
    *EMOTIONS,
    *VAD_DIMS,
]


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_pasted_outputs(path: Path, allow_duplicate_batch_sections: bool = False) -> list[tuple[int, pd.DataFrame]]:
    text = path.read_text(encoding="utf-8").strip()
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty pasted output: {path}")

    batch_starts: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        batch_match = re.match(r"^OG\s+Batch\s+(\d+)\s+output:\s*$", line.strip(), flags=re.IGNORECASE)
        if batch_match:
            batch_starts.append((index, int(batch_match.group(1))))
    if not batch_starts or batch_starts[0][0] != 0:
        raise ValueError("First non-empty line must be like: OG Batch 1 output:")

    parsed: list[tuple[int, pd.DataFrame]] = []
    for start_pos, (line_index, batch_number) in enumerate(batch_starts):
        next_line_index = batch_starts[start_pos + 1][0] if start_pos + 1 < len(batch_starts) else len(lines)
        csv_text = "\n".join(lines[line_index + 1 : next_line_index])
        reader = csv.DictReader(io.StringIO(csv_text))
        if reader.fieldnames != OUTPUT_COLUMNS:
            raise ValueError(f"Unexpected CSV header for OG batch {batch_number}: {reader.fieldnames}. Expected: {OUTPUT_COLUMNS}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"No CSV rows found after OG batch {batch_number} tracking line.")
        parsed.append((batch_number, pd.DataFrame(rows, columns=OUTPUT_COLUMNS)))

    batch_numbers = [batch_number for batch_number, _ in parsed]
    if not allow_duplicate_batch_sections and len(batch_numbers) != len(set(batch_numbers)):
        raise ValueError(f"Duplicate batch sections found: {batch_numbers}")
    return parsed


def expected_for_batch(manifest: pd.DataFrame, batch_number: int) -> pd.DataFrame:
    prefix = f"og784_batch_{batch_number:02d}_"
    out = manifest[manifest["batch_file"].astype(str).str.startswith(prefix)].copy()
    if out.empty:
        raise ValueError(f"No manifest rows found for OG batch {batch_number}.")
    return out


def validate_rows(df: pd.DataFrame, expected: pd.DataFrame, batch_number: int, allow_partial: bool = False) -> pd.DataFrame:
    if allow_partial:
        required_missing = df[OUTPUT_COLUMNS].isna() | (df[OUTPUT_COLUMNS].astype(str) == "")
        incomplete_mask = required_missing.any(axis=1)
        if incomplete_mask.any():
            incomplete_indexes = df.index[incomplete_mask].tolist()
            if incomplete_indexes != [df.index[-1]]:
                raise ValueError(f"OG batch {batch_number} contains incomplete non-trailing rows: {incomplete_indexes}")
            dropped_item_id = df.loc[df.index[-1], "item_id"]
            print(f"Dropping incomplete trailing row from OG batch {batch_number}: {dropped_item_id}")
            df = df.iloc[:-1].copy()

    if not allow_partial and len(df) != len(expected):
        raise ValueError(f"OG batch {batch_number} has {len(df)} rows; expected {len(expected)}.")
    if allow_partial and len(df) > len(expected):
        raise ValueError(f"OG batch {batch_number} has {len(df)} rows; expected at most {len(expected)}.")
    if not df["item_id"].is_unique:
        raise ValueError(f"OG batch {batch_number} contains duplicate item_id values.")

    expected_ids = expected["item_id"].astype(str).tolist()
    actual_ids = df["item_id"].astype(str).tolist()
    missing = [value for value in expected_ids if value not in set(actual_ids)]
    extra = [value for value in actual_ids if value not in set(expected_ids)]
    if missing or extra:
        if not allow_partial or extra:
            raise ValueError(f"OG batch {batch_number} ID mismatch. Missing={missing}; extra={extra}")
    expected_subset_order = [value for value in expected_ids if value in set(actual_ids)]
    if actual_ids != (expected_subset_order if allow_partial else expected_ids):
        raise ValueError(f"OG batch {batch_number} row order differs from manifest order.")

    expected_lookup = expected.set_index("item_id")
    for _, row in df.iterrows():
        item_id = row["item_id"]
        exp = expected_lookup.loc[item_id]
        for column in ["source_type", "perfume_id", "perfume_name"]:
            if str(row[column]) != str(exp[column]):
                raise ValueError(f"{item_id} has wrong {column}: {row[column]} != {exp[column]}")

    df = df.copy()
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
    return df


def completed_full_batches(combined: pd.DataFrame, manifest: pd.DataFrame) -> int:
    complete = 0
    for batch_file, expected in manifest.groupby("batch_file", sort=True):
        expected_ids = set(expected["item_id"].astype(str))
        actual_ids = set(combined.loc[combined["item_id"].isin(expected_ids), "item_id"].astype(str))
        if actual_ids == expected_ids:
            complete += 1
    return complete


def write_outputs(
    df: pd.DataFrame,
    pasted_path: Path,
    manifest: pd.DataFrame,
    batch_number: int,
    out_dir: Path,
    output_tag: str,
    allow_partial: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_batches"
    batch_dir = out_dir / "batch_csv"
    raw_dir.mkdir(exist_ok=True)
    batch_dir.mkdir(exist_ok=True)

    if allow_partial:
        part_index = len(sorted(raw_dir.glob(f"{output_tag}_batch_{batch_number:03d}_part_*.txt"))) + 1
        raw_path = raw_dir / f"{output_tag}_batch_{batch_number:03d}_part_{part_index:02d}.txt"
    else:
        raw_path = raw_dir / f"{output_tag}_batch_{batch_number:03d}.txt"
    batch_csv_path = batch_dir / f"{output_tag}_batch_{batch_number:03d}.csv"
    combined_csv_path = out_dir / f"{output_tag}.csv"
    pipeline_csv_path = out_dir / f"{output_tag}_pipeline.csv"
    report_path = out_dir / f"{output_tag}_report.json"

    raw_path.write_text(pasted_path.read_text(encoding="utf-8").strip() + "\n", encoding="utf-8")
    if allow_partial and batch_csv_path.exists():
        prior = pd.read_csv(batch_csv_path)
        df = pd.concat([prior, df], ignore_index=True)
        if df["item_id"].duplicated().any():
            duplicated = df.loc[df["item_id"].duplicated(), "item_id"].tolist()
            raise ValueError(f"Batch {batch_number} partial merge contains duplicate item IDs: {duplicated}")
        expected = expected_for_batch(manifest, batch_number)
        expected_order = {item_id: idx for idx, item_id in enumerate(expected["item_id"].astype(str).tolist())}
        df["_batch_order"] = df["item_id"].map(expected_order)
        df = df.sort_values("_batch_order").drop(columns=["_batch_order"]).reset_index(drop=True)
        df = validate_rows(df, expected, batch_number, allow_partial=True)
    df.to_csv(batch_csv_path, index=False)

    batch_frames = [pd.read_csv(path) for path in sorted(batch_dir.glob(f"{output_tag}_batch_*.csv"))]
    combined = pd.concat(batch_frames, ignore_index=True) if batch_frames else df.copy()
    manifest_order = {item_id: idx for idx, item_id in enumerate(manifest["item_id"].astype(str).tolist())}
    combined["_manifest_order"] = combined["item_id"].map(manifest_order)
    combined = combined.sort_values("_manifest_order").drop(columns=["_manifest_order"]).reset_index(drop=True)
    if combined["item_id"].duplicated().any():
        duplicated = combined.loc[combined["item_id"].duplicated(), "item_id"].tolist()
        raise ValueError(f"Combined output contains duplicate item IDs: {duplicated}")

    manifest_meta = manifest[["item_id", "perfume_id_hash"]].copy()
    combined_with_hash = combined.merge(manifest_meta, on="item_id", how="left", validate="one_to_one")
    combined_with_hash.to_csv(combined_csv_path, index=False)
    combined_with_hash[PIPELINE_COLUMNS].to_csv(pipeline_csv_path, index=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_tag": output_tag,
        "last_ingested_batch": batch_number,
        "completed_rows": int(len(combined_with_hash)),
        "completed_batches": int(completed_full_batches(combined_with_hash, manifest)),
        "target_rows": int(len(manifest)),
        "remaining_rows": int(len(manifest) - len(combined_with_hash)),
        "emotion_positive_counts": {column: int(combined_with_hash[column].sum()) for column in EMOTIONS},
        "vad_summary": {
            column: {
                "mean": float(combined_with_hash[column].mean()),
                "min": int(combined_with_hash[column].min()),
                "max": int(combined_with_hash[column].max()),
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
        "note": "Browser Gemini Pro pseudo-labels for frozen OG 784 perfume-level review excerpts.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest OG 784 browser Gemini label output.")
    parser.add_argument("--pasted-text", type=Path, required=True)
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=data_path("annotation_packets", "og784_gemini_browser_pro_review_level", "manifest.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=labels_path("gemini_browser_og784"))
    parser.add_argument("--output-tag", default="gemini_browser_labels_og784_v1")
    parser.add_argument("--allow-partial", action="store_true", help="Allow and merge incomplete browser batch outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv)
    parsed_outputs = parse_pasted_outputs(args.pasted_text, allow_duplicate_batch_sections=args.allow_partial)
    report = None
    for batch_number, df in parsed_outputs:
        expected = expected_for_batch(manifest, batch_number)
        df = validate_rows(df, expected, batch_number, allow_partial=args.allow_partial)
        report = write_outputs(
            df,
            args.pasted_text,
            manifest,
            batch_number,
            args.out_dir,
            args.output_tag,
            allow_partial=args.allow_partial,
        )
        print(f"Ingested OG batch {batch_number}: {len(df)} rows")
    assert report is not None
    print(f"Combined rows: {report['completed_rows']} / {report['target_rows']}")
    print(f"Combined CSV: {report['outputs']['combined_csv']}")
    print(f"Pipeline CSV: {report['outputs']['pipeline_csv']}")


if __name__ == "__main__":
    main()
