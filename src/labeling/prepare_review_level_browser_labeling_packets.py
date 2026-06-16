# Date Created: 2026-06-12
# Date Last Modified: 2026-06-14

"""Prepare browser LLM labeling packets for review-level synthetic data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.labeling.strict_schema import EMOTIONS, VAD_DIMS
from src.utils.paths import PROJECT_ROOT, data_path

OUTPUT_COLUMNS = [
    "synthetic_review_id",
    "source_type",
    "perfume_id",
    "perfume_name",
    "review_index",
    *EMOTIONS,
    *VAD_DIMS,
]

PROMPT_TEXT = """You are labeling perfume review text for an academic thesis.

Task: For each review in the batch, assign emotion-tone labels using ONLY the review text provided.

Rules:
- Label each review independently.
- Do NOT infer from brand, perfume name, reputation, note list, marketing concept, or niche status.
- Labels reflect the reviewer's emotional tone expressed in the text.
- Set an emotion to 1 only when there is clear linguistic evidence. Otherwise set it to 0.
- Multiple emotions can be 1.
- Surprise=1 only when the text explicitly expresses unexpectedness, such as "surprised", "did not expect", "caught me off guard", or similar language.
- Anticipation=1 when the text clearly expresses expectation, looking forward, curiosity, desire to keep testing, or waiting for something to unfold.
- Trust=1 when the text clearly expresses comfort, safety, reliability, attachment, reassurance, or confidence in wearing the perfume.
- Arousal means emotional activation or calmness, not perfume strength or projection.
- Dominance means confidence/control/assertiveness, not perfume loudness unless framed as personal power/control.
- If emotional evidence is weak or neutral, use all emotions=0 and valence=5, arousal=5, dominance=5.

Emotion columns, binary 0/1:
joy,sadness,anger,fear,trust,surprise,anticipation,disgust

VAD columns, integer 1-9:
valence: 1 very negative, 5 neutral, 9 very positive
arousal: 1 very calm, 5 neutral, 9 highly activated
dominance: 1 passive/gentle, 5 neutral, 9 commanding/confident

Return no markdown, no explanation, and no code fences.
Apart from the batch tracking line requested below, return only the CSV header and rows.
Use exactly this CSV header:
synthetic_review_id,source_type,perfume_id,perfume_name,review_index,joy,sadness,anger,fear,trust,surprise,anticipation,disgust,valence,arousal,dominance

Return exactly one output row per input review.
"""


def clean_text(value: object, max_chars: int) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def chunks(items: Sequence[dict], size: int) -> list[Sequence[dict]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def require_columns(df: pd.DataFrame, path: Path) -> None:
    required = {"synthetic_review_id", "perfume_id", "Perfume", "Final Transcript"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def write_batch(path: Path, batch: Sequence[dict], batch_label: str) -> None:
    lines = [
        PROMPT_TEXT.strip(),
        "",
        "RESPONSE FORMAT",
        f"Start your response with exactly this tracking line: {batch_label} output:",
        "Then immediately output the CSV header and one CSV row per input review.",
        "",
        "INPUT BATCH",
        "",
    ]
    for item in batch:
        lines.extend(
            [
                "BEGIN_REVIEW",
                f"synthetic_review_id: {item['synthetic_review_id']}",
                "source_type: synthetic",
                f"perfume_id: {item['perfume_id']}",
                f"perfume_name: {item['perfume_name']}",
                f"review_index: {item['review_index']}",
                f"review_text: {item['review_text']}",
                "END_REVIEW",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_manifest(path: Path, items: Sequence[dict]) -> None:
    fieldnames = [
        "synthetic_review_id",
        "source_type",
        "perfume_id",
        "perfume_name",
        "review_index",
        "text_word_count",
        "text_char_count",
        "batch_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({field: item.get(field, "") for field in fieldnames})


def write_output_template(path: Path, items: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "synthetic_review_id": item["synthetic_review_id"],
                    "source_type": "synthetic",
                    "perfume_id": item["perfume_id"],
                    "perfume_name": item["perfume_name"],
                    "review_index": item["review_index"],
                    **{column: "" for column in EMOTIONS + VAD_DIMS},
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare review-level browser LLM labeling packets.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "synthetic_review_level_tests"
        / "emotion_rich_v2_pilot100"
        / "annotation"
        / "review_level_annotation_template.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=data_path("annotation_packets", "emotion_rich_v2_pilot100_review_level"),
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=1400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    require_columns(df, args.input_csv)
    if not df["synthetic_review_id"].astype(str).is_unique:
        raise ValueError("synthetic_review_id must be unique.")

    items: list[dict] = []
    for _, row in df.iterrows():
        text = clean_text(row["Final Transcript"], args.max_chars)
        review_index = row.get("review_index", "")
        if pd.isna(review_index) or review_index == "":
            review_index = str(row["synthetic_review_id"]).rsplit("-", 1)[-1]
        item = {
            "synthetic_review_id": str(row["synthetic_review_id"]),
            "source_type": "synthetic",
            "perfume_id": str(row["perfume_id"]),
            "perfume_name": str(row["Perfume"]),
            "review_index": int(review_index) if str(review_index).isdigit() else review_index,
            "review_text": text,
            "text_word_count": len(re.findall(r"\b\w+\b", text)),
            "text_char_count": len(text),
        }
        items.append(item)

    for batch_index, batch in enumerate(chunks(items, args.batch_size), start=1):
        first = (batch_index - 1) * args.batch_size + 1
        last = first + len(batch) - 1
        batch_file = f"v2_review_batch_{batch_index:02d}_reviews_{first:03d}-{last:03d}.txt"
        batch_label = f"Batch {batch_index}"
        for item in batch:
            item["batch_file"] = batch_file
        write_batch(args.out_dir / batch_file, batch, batch_label)

    write_manifest(args.out_dir / "manifest.csv", items)
    write_output_template(args.out_dir / "output_template.csv", items)
    (args.out_dir / "prompt.md").write_text(PROMPT_TEXT, encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": stringify_path(args.input_csv),
        "out_dir": stringify_path(args.out_dir),
        "review_rows": len(items),
        "unique_perfumes": int(df["perfume_id"].nunique()),
        "batch_size": args.batch_size,
        "batch_count": len(chunks(items, args.batch_size)),
        "output_columns": OUTPUT_COLUMNS,
        "note": "These packets contain no labels. Paste each batch into a browser LLM and save the returned CSV rows.",
    }
    (args.out_dir / "packet_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    readme = f"""# Emotion-Rich V2 Review-Level Labeling Packets

Paste one batch file at a time into the browser LLM.

- Batch files: `v2_review_batch_*.txt`
- Prompt only: `prompt.md`
- Manifest: `manifest.csv`
- Empty output template: `output_template.csv`

Each batch asks the browser LLM to start with `Batch N output:` and then return the CSV rows.
You can paste that whole response back into Codex for validation and file writing.

Expected output schema:

```csv
{",".join(OUTPUT_COLUMNS)}
```

When all batches are labeled, combine the returned CSV rows into one file with the same header.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote {len(items)} review items to {args.out_dir}")
    print(f"Batch files: {len(chunks(items, args.batch_size))}")
    print(f"Output template: {args.out_dir / 'output_template.csv'}")


if __name__ == "__main__":
    main()
